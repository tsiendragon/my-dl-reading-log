# A Practical Introduction to Multimodal Large Language Models

## Big-picture takeaways

- **Research focus is shifting from “training everything from scratch” to “connecting strong single-modality backbones”**: use a powerful LLM as the “brain,” reuse mature vision/audio encoders as the “senses,” and focus on aligning heterogeneous modalities so they can reason together—cutting cost and improving efficiency.
- **Two-stage training paradigm**: first perform **multimodal pretraining (MM-PT)** to align modalities with the LLM space; then perform **multimodal instruction tuning (MM-IT)** (SFT and possibly RLHF) to align with human intent and improve interaction and generalization.

## Core architecture

1. **Modality encoders**: extract features for images/videos/audio (e.g., ViT, CLIP/EVA-CLIP, HuBERT, Whisper)
2. **Input projector**: map non-text features into the LLM token space; implementations range from **MLP** to **Cross-Attention, Q-Former, P-Former**, etc.
3. **LLM backbone**: performs understanding/reasoning/decision-making and produces text tokens as well as “signal tokens” that can control other modality generators; commonly adapted with **PEFT (Prefix/Adapter/LoRA)**.

4. **Output projector**: maps LLM outputs into conditional representations consumable by downstream generators.

5. **Modality generators**: call existing diffusion/decoder models (e.g., **Stable Diffusion / Zeroscope / AudioLDM-2**) to generate via conditional denoising

> Summary: a typical “lightly connect, heavily reuse” engineering pattern—freeze as many large pretrained blocks as possible; focus optimization on input/output projectors and a few adapters. When pushing the upper bound, fine-tune encoders and/or the LLM.

## Training pipeline and data

- **Pretraining (MM-PT)**: use large-scale X-Text (image/video/audio–text) to align each modality with the LLM space.
- **Instruction tuning (MM-IT)**: convert PT data into instruction format for SFT; optionally add RLHF to improve instruction-following, zero-shot, and dialogue capabilities.
- **Evaluation and data**: a systematic review of the mainstream ~18 vision-language benchmarks and data construction methods, including practical training tips that matter in practice.

## Representative models and highlights (selected)

- **Flamingo, BLIP-2, LLaVA, MiniGPT-4**: from cross-modal in-context learning to lightweight Q-Former and open MM-IT data/benchmarks—cementing the “connect–align–instructionalize” paradigm.
- **mPLUG-Owl, X-LLM, VideoChat, InstructBLIP**: modular training, audio/video extension, chat-centric video understanding, and instruction-aware features by only tuning Q-Former.
- **PandaGPT, PaLI-X, Video-LLaMA, Video-ChatGPT, Shikra, DLP (P-Former)**: any-to-any interfaces, mixed-objective pretraining, spatiotemporal representations, precise referring and prompt prediction driven by single-modality data.
- **BuboGPT, ChatSpot, Qwen-VL, NExT-GPT, MiniGPT-5**: fine-grained cross-modal semantics, region-level interaction, multi-language/multi-image input, end-to-end any-to-any, and vokens/classifier-free guidance.

## Outlook (original viewpoints preserved)

- **Five directions**: stronger models, **more challenging evaluation**, **on-device/lightweight deployment**, **embodied intelligence**, and **continuous instruction alignment**.


### One-page pocket checklist

- **Paradigm**: encoders + projectors (in/out) + LLM + generators → align first, then instruction-tune.
- **Key connectors**: Cross-Attn / **Q-Former / P-Former**; **PEFT** for low-cost adaptation.
- **Strategy**: reuse strong single-modality models and strong LLMs; keep connections light; fine-tune encoders/LLMs when chasing the upper bound.
- **Representative works**: BLIP-2 / LLaVA / MiniGPT-4 / Qwen-VL / NExT-GPT (any-to-any).
- **Evaluation/data**: curated overview of ~18 VL benchmarks and data construction methods—good as an onboarding roadmap.

> Note: the original Zhihu link may not be directly accessible in this environment; the above is distilled from a full reprint page (which includes attribution to the original post), with wording adjusted minimally to preserve intent.

## How alignment modules work (projectors and connectors)

- **MLP projector**: simplest mapping from encoder features to LLM token embeddings; cheap, widely used for fast prototyping.
- **Cross-Attention projector**: lets the LLM attend to modality features directly; better capacity, higher compute cost.
- **Q-Former / P-Former**: query-based or prompt-based lightweight transformers that compress modality features into a handful of “learned tokens,” balancing information retention with token budget.
- **Design choices**: number of visual/audio tokens, where to inject (prepend vs interleave), positional encodings for grids/patches, and whether to freeze encoder and/or LLM.

## Training objectives and practical recipes

- **Contrastive alignment** (InfoNCE/CLIP-style) for coarse matching.
- **Captioning/denoising** (autoregressive or masked) for fine-grained grounding.
- **Instruction-following dialogue** for conversational grounding across modalities.
- **Region-level supervision** (boxes/masks) when precise grounding is required.
- **Preference optimization** (DPO/RLHF variants) for helpfulness, harmlessness, and reduced hallucination.

Typical lightweight recipe: start from a strong vision encoder and a mid-size LLM, train a projector with mixed caption + VQA + OCR-style data; optionally unfreeze higher encoder blocks or add LoRA on the LLM to push quality.

## 8) Inference and prompting

- **Visual tokenization**: flatten grid features into tokens; use special markers (<image>, region tags) to delineate modality spans.
- **Prompt templates**: instruction-style prompts (“You are a multimodal assistant…”) plus image placeholders; chain-of-thought can be encouraged via system prompts.
- **Multi-image and long context**: interleave multiple image tokens with textual context; for videos, chunk frames and pool temporally (mean/attn pooling) before feeding the LLM.
- **Tool-use via signal tokens**: the LLM can emit control tokens to trigger external generators (e.g., image editing or TTS) through the output projector.

## 9) Evaluation landscape (by capability)

- **VQA / Grounding**: general question answering, spatial reasoning, and referring expressions.
- **Captioning / OCR**: dense descriptions, scene text understanding, charts/diagrams.
- **Reasoning**: math and science diagrams, multi-hop visual-text reasoning.
- **Video understanding**: temporal reasoning, event localization, and audio-visual cues.
- **Safety**: jailbreak robustness, sensitive content handling, factuality and groundedness.

When comparing models, consider input resolution, token budget, context length, and whether external OCR/ASR is used.

## 10) Deployment and performance engineering

- **Quantization** (e.g., weight-only or AWQ/GPTQ-style), **KV-cache** management, and **paged attention** to reduce memory and latency.
- **Encoder sharing** across turns and **lazy decoding** to avoid recomputing visual features.
- **Streaming** outputs and **chunked video** ingestion to keep latency predictable.
- **On-device** variants use smaller LLMs, lighter encoders, and fewer visual tokens.

## 11) Safety and ethics

- Guard against **in-image prompt injection/jailbreak**, sensitive content, and PII leakage.
- Reduce **hallucination** via better grounding data, preference optimization, and constrained decoding.
- Provide user controls, audit logs, and clear failure modes.

## 12) Open problems and 2025 trends

- **Any-to-any**: unified interfaces across text, image, audio, and video with consistent controllability.
- **Data/compute efficiency**: stronger results with fewer tuned parameters and smarter curricula.
- **Multilingual multimodality**: robust performance across languages and scripts (incl. dense OCR).
- **Agents and embodiment**: tool-use, planning, and interactive perception-action loops.
- **Edge and mobile**: low-latency, privacy-preserving deployments.

## 13) Starter resources (open-source examples)

- Model families: LLaVA, Qwen-VL, BLIP-2/InstructBLIP, InternVL, Idefics2, MiniGPT series.
- Common datasets: image–text (COCO/CC3M/CC12M/LAION subsets), instruction datasets (LLaVA-150K-style), OCR/ChartQA, VQA, and curated video–text pairs.
