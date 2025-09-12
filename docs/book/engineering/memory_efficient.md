# Memory-Efficient Deep Learning: Key Techniques Explained

Training modern deep learning models is expensive — they eat up GPU memory fast. Fortunately, there are techniques that help us train larger models faster while consuming fewer resources. Let’s break down some of the most important ones.

---

## 1. Data Parallelism (DP) vs Distributed Data Parallel (DDP) vs Fully Sharded Data Parallel (FSDP)

| Method   | How it works                                                                                                       | Pros                                                 | Cons                                        |
| -------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------- |
| **DP**   | Splits input data across GPUs, but keeps a full copy of the model on each GPU.                                     | Easy to implement, simple                            | High memory usage (model fully replicated)  |
| **DDP**  | Each GPU runs a copy of the model and computes gradients on its shard of data. Gradients are averaged across GPUs. | Efficient, scales well, widely used                  | Still replicates the full model on each GPU |
| **FSDP** | Splits both the **model parameters** and **gradients** across GPUs, only loading/sharing shards when needed.       | Huge memory savings, enables training massive models | More complex setup, communication overhead  |

### DP
**Data Parallel (DP) — Principle, Steps, Pros/Cons**
![dp illus](https://miro.medium.com/v2/resize:fit:720/format:webp/1*Kc3ZE6C6V_gF1VKXpdaDtg.png)
#### Core idea

* **Replicate the full model on every GPU.**
* **Split the input batch** into mini-batches, dispatch one per GPU for forward & backward.
* **Gather gradients on device\[0]**, update there, **broadcast updated weights** back to other GPUs.

#### One training iteration (matches 1/2/3 in the diagram)

1. **Parallel compute (red):**
   Each GPU runs forward → loss → backward, producing its local gradient $g_i$.
2. **Gradient aggregation (light blue):**
   All $g_i$ are copied to **device\[0]** and reduced (sum/mean) to global gradient $g$.
3. **Update & broadcast (green):**
   On **device\[0]**, update parameters $\theta \leftarrow \theta - \eta g$;
   broadcast the new $\theta$ to all other GPUs.

#### Pseudocode (simplified)

```python
# model replicated on each GPU
grads = [backward(model_i(x_i)) for i in gpus]  # 1
g = reduce_to_device0(grads, op="mean")         # 2
theta0 = optimizer_update(theta0, g)            # 3
broadcast(theta0, to=gpus[1:])
```

#### Advantages

* Very easy to use (PyTorch `DataParallel`), minimal code changes.
* Leverages multiple GPUs for forward/backward in parallel.

#### Limitations / Pitfalls

* **High memory**: full model copy on every GPU.
* **Bottleneck on device\[0]**: centralized gradient reduction & update.
* Extra **tensor copies** → bandwidth-sensitive; single-process/GIL overhead.
* In practice, **slower and less scalable than DDP**.




## 2. Low-Precision Training: FP4 & FP8 Quantization

Instead of using standard FP32 (32-bit floats), models can run with fewer bits per number:

| Format          | Bits | Memory Saved  | Use Case                                        |
| --------------- | ---- | ------------- | ----------------------------------------------- |
| **FP16 / BF16** | 16   | \~50% vs FP32 | Today’s standard for training                   |
| **FP8**         | 8    | \~75% vs FP32 | Training & inference, NVIDIA H100 supports this |
| **FP4**         | 4    | \~87% vs FP32 | Mostly research/prototype, very aggressive      |

Lower precision = smaller tensors = less GPU memory + faster compute.
Modern hardware (like NVIDIA H100) makes FP8 training practical. FP4 is still experimental, but very promising.

---

## 3. Gradient Checkpointing
Backprop needs intermediate **activations** from the forward pass. Storing every activation is memory-heavy. **Gradient checkpointing** saves only a subset (“checkpoints”) and **recomputes** the missing activations during backward. You trade extra compute for much lower memory.



### Example

Let we have a network,  and $x$ be the input, $y$ be the output, and $\hat{y}$ is our prediction. Let's see how the gradient is calculated.
![alt text](../../images/image-7.png)
* $a_1 = w_1 x$, $a_2 = w_2 a_1$, $a_3 = w_3 a_2$, $\hat{y} = w_4 a_3$
* Loss $\displaystyle \mathcal{L}=\tfrac12(\hat{y}-y)^2$

**Gradients (correct sign uses $\hat{y}-y$):**

$$
\frac{\partial \mathcal{L}}{\partial w_4}=(\hat{y}-y)\,a_3,\quad
\frac{\partial \mathcal{L}}{\partial w_3}=(\hat{y}-y)\,w_4\,a_2,
$$

$$
\frac{\partial \mathcal{L}}{\partial w_2}=(\hat{y}-y)\,w_4\,w_3\,a_1,\quad
\frac{\partial \mathcal{L}}{\partial w_1}=(\hat{y}-y)\,w_4\,w_3\,w_2\,x.
$$

**Naïve training:** store all $a_1,a_2,a_3$ during forward → large memory.

**Checkpointing idea:** keep only a few (e.g., $a_1,a_3$), drop $a_2$. During backward, **recompute** $a_2=w_2 a_1$ on the fly to form the gradients above.


### What you gain vs. pay

| Aspect             | No checkpointing          | With checkpointing                                   |
| ------------------ | ------------------------- | ---------------------------------------------------- |
| Activations stored | All ($a_1,a_2,a_3,\dots$) | Only chosen checkpoints (e.g., $a_1,a_3$)            |
| GPU memory         | High                      | **30–70% lower** (rule of thumb)                     |
| Compute (time)     | Baseline                  | **Higher** (extra forward recomputation in backward) |
| Best for           | Small/medium nets         | Deep/large models (Transformers)                     |



### How it’s implemented (PyTorch)

**Granular control**

```python
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class Block(nn.Module):
    def __init__(self, f):  # f is any submodule
        super().__init__()
        self.f = f
    def forward(self, x):
        # this block will be recomputed in backward
        return checkpoint(self.f, x)

# Example: wrap deep stacks with checkpointed blocks
```

**Transformers/Trainer switch**

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,   # turn it on
    fp16=True,                     # often used together
)
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
```


### Practical rules for Activations to checkpoint

1. **Transformer (best default)**

   * **Checkpoint per block**: keep the *block input*; recompute inside (MH-Attn + MLP).
     Savings 30–50% with small speed hit.
   * Still OOM? **Checkpoint sublayers**: (a) attention, (b) MLP separately; keep residuals.
   * Very long seq: pair with **flash/efficient attention**; checkpoint the attention path first.

2. **CNN/ViT hybrids**

   * Early layers have huge H×W → **checkpoint early stages** (pre-downsample convs).
     Keep tensors *after* downsampling (they’re small), drop pre-pool/stride-2 outputs.

3. **“√N” rule for chains**

   * For N similar layers, choose \~√N checkpoints spaced evenly (Revolve-style).
     In PyTorch: `checkpoint_sequential(modules, segments=int(math.sqrt(N)))`.

4. **Prefer dropping**

   * Activations with **large bytes** (batch×seq×hidden or big feature maps).
   * **Pure, stateless** computations (GELU, Linear, LayerNorm, convs without data-dependent control).

5. **Prefer keeping**

   * **Layer inputs at boundaries** (so you can recompute inside the segment).
   * Activations that are **very expensive** to recompute (e.g., custom heavy ops) or **non-deterministic** unless seeded (dropout is fine if seeded).
   * Anything mutated **in-place** (avoid in-place inside checkpointed segments).

6. **Profile, then decide**

   * Use `torch.cuda.memory_allocated()` / `torch.profiler` to find top memory tensors; checkpoint those paths first.



## Final Thoughts

* **FSDP** lets you scale models beyond single-GPU memory limits.
* **FP8 / FP4** push efficiency by squeezing numbers into fewer bits.
* **Gradient checkpointing** trades a little compute for big memory savings.

Together, these techniques make it possible to train models that used to be “too big for your GPU.” If you’re pushing limits, these are must-know tools.

---

Do you want me to also make a **diagram (Mermaid or simple schematic)** comparing these methods visually for your blog?
