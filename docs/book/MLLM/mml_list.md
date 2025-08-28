# 多模态大模型总览（2020–2025）

本文档整理了2020年至2025年间发布的主要多模态大模型，涵盖OpenAI、Google、Meta、阿里巴巴、智谱AI、上海AI Lab等主流机构的代表性工作。

## 多模态大模型发展时间线

### 2020年
- **DALL-E** (OpenAI) - 首个大规模文本到图像生成模型
- **CLIP** (OpenAI) - 图像-文本对比学习的突破性工作

### 2021年
- **SimVLM** (Google) - 简单视觉语言模型预训练
- **DALL-E 2** (OpenAI) - 改进的文本到图像生成

### 2022年
- **Flamingo** (DeepMind) - 少样本学习的视觉语言模型
- **BLIP** (Salesforce) - 双向语言图像预训练
- **PaLI** (Google) - 联合缩放的多语言语言-图像模型

### 2023年
- **GPT-4V** (OpenAI) - 具备视觉能力的GPT-4
- **BLIP-2** (Salesforce) - 引入Q-Former的改进版本
- **LLaVA** (University of Wisconsin-Madison) - 大型语言和视觉助手
- **MiniGPT-4** (King Abdullah University) - 轻量级多模态模型
- **InstructBLIP** (Salesforce) - 指令感知的视觉语言模型
- **Qwen-VL** (阿里巴巴) - 千问视觉语言模型
- **IDEFICS** (Hugging Face) - 开源视觉语言模型
- **InternLM-XComposer** (上海AI Lab) - 视觉语言大模型
- **ERNIE-ViLG 2.0** (百度) - 知识增强的文本到图像扩散模型

### 2024年
- **SPHINX-X** - 多模态大语言模型系列的扩展
- **Gato** (DeepMind) - 通用智能体

## 详细模型信息表

| 模型 | 发布时间 | 视觉编码器 | 转换器架构 | 语言模型(LLM) | 输入分辨率 | 模型大小 | 所属机构 | 主要特点 | 论文/链接 |
|------|----------|------------|------------|---------------|------------|----------|----------|----------|----------|
| [CLIP](https://openai.com/index/clip/) | 2021.01 | ViT-B/32, ViT-L/14 | Vision Transformer | - | 224×224, 336×336 | 63M-427M | OpenAI | 图像-文本对比学习的开创性工作 | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) |
| [DALL-E](https://openai.com/index/dall-e/) | 2021.01 | - | Transformer | GPT-3 变体 | 256×256 | 12B | OpenAI | 首个大规模文本到图像生成模型 | [arXiv:2102.12092](https://arxiv.org/abs/2102.12092) |
| [SimVLM](https://research.google/blog/simvlm-simple-visual-language-model-pre-training-with-weak-supervision/) | 2021.08 | ResNet, ViT | - | PrefixLM | 224×224 | 630M-4.6B | Google | 弱监督预训练的简单视觉语言模型 | [arXiv:2108.10904](https://arxiv.org/abs/2108.10904) |
| [DALL-E 2](https://openai.com/index/dall-e-2/) | 2022.04 | CLIP ViT-H/14 | - | 修改版GPT-3 | 1024×1024 | - | OpenAI | 改进的文本到图像生成，使用unCLIP | [arXiv:2204.06125](https://arxiv.org/abs/2204.06125) |
| [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model) | 2022.04 | NFNet | Perceiver Resampler | Chinchilla | 320×320 | 3B-80B | DeepMind | 少样本上下文学习能力 | [arXiv:2204.14198](https://arxiv.org/abs/2204.14198) |
| [BLIP](https://github.com/salesforce/BLIP) | 2022.05 | ViT-B/16 | - | BERT | 224×224, 384×384 | 129M-579M | Salesforce | 双向语言图像预训练 | [arXiv:2201.12086](https://arxiv.org/abs/2201.12086) |
| [PaLI](https://ai.googleblog.com/2022/09/pali-scaling-language-image-learning-in.html) | 2022.09 | ViT-e | - | UL2 | 588×588 | 17B | Google | 多语言视觉语言模型 | [arXiv:2209.06794](https://arxiv.org/abs/2209.06794) |
| [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | 2023.01 | EVA-CLIP ViT-g | Q-Former | OPT, FlanT5 | 224×224 | 2.7B-11B | Salesforce | 引入Q-Former进行模态对齐 | [arXiv:2301.12597](https://arxiv.org/abs/2301.12597) |
| [LLaVA](https://llava-vl.github.io/) | 2023.04 | CLIP ViT-L/14 | Linear Projection | Vicuna | 224×224 | 7B-13B | UW-Madison等 | 视觉指令跟随能力 | [arXiv:2304.08485](https://arxiv.org/abs/2304.08485) |
| [MiniGPT-4](https://minigpt-4.github.io/) | 2023.04 | EVA-CLIP ViT-g | Linear Projection | Vicuna | 224×224 | 7B-13B | KAUST | 轻量级设计，强对话能力 | [arXiv:2304.10592](https://arxiv.org/abs/2304.10592) |
| [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) | 2023.05 | EVA-CLIP ViT-g | Q-Former | Vicuna, FlanT5 | 224×224 | 7B-11B | Salesforce | 指令感知的视觉语言理解 | [arXiv:2305.06500](https://arxiv.org/abs/2305.06500) |
| [GPT-4V](https://openai.com/research/gpt-4v-system-card) | 2023.09 | - | - | GPT-4 | - | - | OpenAI | 多模态版GPT-4 | [GPT-4V System Card](https://cdn.openai.com/papers/GPTV_System_Card.pdf) |
| [Qwen-VL](https://github.com/QwenLM/Qwen-VL) | 2023.08 | OpenCLIP ViT-bigG | Cross-Attention | Qwen-7B | 448×448 | 7B-72B | 阿里巴巴 | 支持多语言和高分辨率输入 | [arXiv:2308.12966](https://arxiv.org/abs/2308.12966) |
| [IDEFICS](https://huggingface.co/blog/idefics) | 2023.08 | OpenCLIP ViT-H | Perceiver Resampler | LLaMA | 224×224 | 9B-80B | Hugging Face | 开源Flamingo复现 | [HuggingFace Blog](https://huggingface.co/blog/idefics) |
| [InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer) | 2023.09 | EVA-CLIP ViT-g | Partial LoRA | InternLM | 224×224 | 7B | 上海AI Lab | 图文创作能力 | [arXiv:2309.15112](https://arxiv.org/abs/2309.15112) |
| [ERNIE-ViLG 2.0](https://wenxin.baidu.com/ernie-vilg) | 2023.06 | - | - | ERNIE 3.0 | 1024×1024 | 24B | 百度 | 知识增强的文本到图像生成 | [arXiv:2210.15257](https://arxiv.org/abs/2210.15257) |
| [SPHINX-X](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/SPHINX) | 2024.02 | CLIP ViT | MLP | LLaMA2, Mixtral | 224×224 | 7B-8×7B | - | 多模态大语言模型系列 | [arXiv:2402.05935](https://arxiv.org/abs/2402.05935) |

## 技术发展趋势

### 1. 架构演进
- **早期（2020-2021）**: 独立设计的端到端模型
- **中期（2022-2023）**: 模块化设计，复用预训练组件
- **现在（2024+）**: 轻量连接器 + 强大基础模型

### 2. 训练范式
- **阶段1**: 联合预训练所有模块
- **阶段2**: 冻结预训练模型 + 训练连接器
- **阶段3**: 多阶段训练（对齐→指令微调）

### 3. 能力扩展
- **理解**: 图像描述 → 复杂推理 → 多轮对话
- **生成**: 单一模态 → 跨模态 → Any-to-Any
- **交互**: 静态问答 → 动态对话 → 具身智能

## 主要机构贡献

### OpenAI
- CLIP, DALL-E系列, GPT-4V
- 引领图像-文本对比学习和文本到图像生成

### Google/DeepMind
- SimVLM, PaLI, Flamingo, Gato
- 在多语言和少样本学习方面领先

### Salesforce
- BLIP系列, InstructBLIP
- Q-Former架构的重要贡献

### 中国机构
- **阿里巴巴**: Qwen-VL系列
- **百度**: ERNIE-ViLG系列
- **上海AI Lab**: InternLM-XComposer
- **清华/智谱**: ChatGLM-6B多模态版本

## 参考资料

1. [CLIP: Connecting text and images | OpenAI](https://openai.com/index/clip/)
2. [DALL·E: Creating images from text | OpenAI](https://openai.com/index/dall-e/)
3. [SimVLM: Simple Visual Language Model Pre-training](https://research.google/blog/simvlm-simple-visual-language-model-pre-training-with-weak-supervision/)
4. [A Survey of State of the Art Large Vision Language Models](https://arxiv.org/html/2501.02189v5)
5. [Introducing IDEFICS: An Open Reproduction of State-of-the-art Visual Language Model](https://huggingface.co/blog/idefics)
6. [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)
7. [Alibaba Cloud launches Qwen-VL](https://technode.com/2023/08/28/alibaba-cloud-launches-open-source-large-vision-language-model-qwen-vl/)

---

*最后更新: 2025年1月*
