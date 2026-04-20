#  Hindi ↔ English Neural Machine Translator

> A Transformer trained from scratch for Hindi ↔ English translation using an Encoder-Decoder architecture, SentencePiece tokenization, and the AI4Bharat / OPUS-100 dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)

---

##  Project Overview

This project implements a full **sequence-to-sequence Transformer** for Hindi ↔ English translation — built from scratch in PyTorch, without using HuggingFace pretrained weights. The goal is to deeply understand how encoder-decoder attention, positional encoding, and cross-attention work in practice.

**Why Hindi ↔ English?**
- Structurally very different languages (SOV vs SVO word order)
- Rich parallel corpora available (AI4Bharat, OPUS-100)
- Directly useful for real-world applications in India

---

##  Architecture

```
Hindi Input  →  SentencePiece Tokenizer  →  Embedding + Positional Encoding
                                                         ↓
                                              Encoder Stack (N=4 layers)
                                              ├── Multi-Head Self-Attention
                                              ├── Feed-Forward Network
                                              └── Layer Norm + Residual

                                                         ↓ (cross-attention)

                                              Decoder Stack (N=4 layers)
                                              ├── Masked Multi-Head Self-Attention
                                              ├── Cross-Attention (encoder output)
                                              ├── Feed-Forward Network
                                              └── Layer Norm + Residual

                                                         ↓
                                              Linear + Softmax  →  English Output
```

| Hyperparameter     | Value            |
|--------------------|------------------|
| `d_model`          | 256              |
| `num_heads`        | 8                |
| `num_layers`       | 4 (enc + dec)    |
| `d_ff`             | 1024             |
| `dropout`          | 0.1              |
| `max_seq_len`      | 128              |
| `vocab_size`       | 32,000 (shared)  |
| `batch_size`       | 64               |
| `warmup_steps`     | 4,000            |

---

##  Repository Structure

```
hindi-english-transformer/
│
├── data/
│   ├── raw/                    # Raw parallel corpus (gitignored)
│   ├── processed/              # Tokenized & binarized data
│   └── download.sh             # Script to fetch OPUS-100 / AI4Bharat
│
├── tokenizer/
│   ├── train_tokenizer.py      # Train SentencePiece on combined corpus
│   └── vocab/                  # Saved .model and .vocab files
│
├── model/
│   ├── attention.py            # Multi-head attention + masking
│   ├── encoder.py              # Encoder layer + stack
│   ├── decoder.py              # Decoder layer + stack
│   ├── embeddings.py           # Token + positional embeddings
│   ├── transformer.py          # Full encoder-decoder model
│   └── utils.py                # Masks, padding helpers
│
├── training/
│   ├── train.py                # Main training loop
│   ├── scheduler.py            # Noam (warmup) learning rate scheduler
│   ├── dataset.py              # PyTorch Dataset + DataLoader
│   └── label_smoothing.py      # Label-smoothed cross-entropy loss
│
├── evaluation/
│   ├── evaluate.py             # BLEU score evaluation
│   └── translate.py            # Inference + beam search
│
├── checkpoints/                # Saved model weights (gitignored)
├── configs/
│   └── base.yaml               # All hyperparameters in one place
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_tokenizer_analysis.ipynb
│   └── 03_attention_visualization.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Getting Started

### 1. Clone and install

```bash
git clone https://github.com/Rajat25022005/hindi-english-transformer.git
cd hindi-english-transformer
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
bash data/download.sh
```

This fetches the **OPUS-100** Hindi-English split (~1M sentence pairs). For a larger dataset, the script also supports AI4Bharat's IndicCorp.

### 3. Train the SentencePiece tokenizer

```bash
python tokenizer/train_tokenizer.py \
  --input data/raw/train.hi data/raw/train.en \
  --model_prefix tokenizer/vocab/hi_en \
  --vocab_size 32000 \
  --model_type bpe
```

A **shared vocabulary** across both languages helps the model leverage overlapping subwords (numbers, proper nouns, borrowed words).

### 4. Preprocess data

```bash
python training/dataset.py --split train
python training/dataset.py --split val
python training/dataset.py --split test
```

### 5. Train

```bash
python training/train.py --config configs/base.yaml
```

Training logs to `wandb` by default. To disable:

```bash
python training/train.py --config configs/base.yaml --no_wandb
```

### 6. Evaluate (BLEU)

```bash
python evaluation/evaluate.py \
  --checkpoint checkpoints/best.pt \
  --split test
```

### 7. Translate

```bash
python evaluation/translate.py \
  --checkpoint checkpoints/best.pt \
  --text "बिल्ली छत पर बैठी थी।" \
  --direction hi2en \
  --beam_size 5
```

---

##  Dataset

| Source        | Pairs      | Notes                              |
|---------------|------------|------------------------------------|
| OPUS-100      | ~1M        | Filtered, high quality             |
| AI4Bharat     | ~2.5M      | Broader domain coverage            |
| FLORES-200    | 1,012      | Evaluation benchmark               |

Data preprocessing steps:
- Remove sentence pairs longer than 128 tokens
- Filter pairs where length ratio > 2.5
- Lowercase English; preserve Hindi script
- Shuffle and split: 98% train / 1% val / 1% test

---

##  Training Details

**Loss:** Label-smoothed cross-entropy (`ε = 0.1`) — prevents the model from becoming overconfident on the training set.

**Optimizer:** AdamW with the Noam scheduling from "Attention Is All You Need":

```
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))
```

**Hardware:** ~6 hours on a single NVIDIA T4 (Google Colab) for 10 epochs on the OPUS-100 subset.

**Target BLEU:** ~28–32 on the FLORES-200 Hi→En benchmark.

---

##  Attention Visualization

The `notebooks/03_attention_visualization.ipynb` notebook lets you visualize which source tokens the decoder attends to when generating each output token.

```python
from model.transformer import Transformer
from evaluation.translate import visualize_attention

visualize_attention(
    model=model,
    src="मुझे भारत से प्यार है।",
    direction="hi2en"
)
```

---

##  Requirements

```
torch>=2.0
sentencepiece>=0.1.99
sacrebleu>=2.3
wandb
pyyaml
tqdm
numpy
matplotlib
```

Install all:

```bash
pip install -r requirements.txt
```

---

##  Roadmap

- [x] Encoder-Decoder Transformer architecture
- [x] SentencePiece shared tokenizer
- [x] Label-smoothed loss + Noam scheduler
- [x] Beam search inference
- [ ] English → Hindi direction (bidirectional training)
- [ ] Byte-Pair Encoding ablation study
- [ ] Attention head pruning experiment
- [ ] Export to ONNX for inference
- [ ] Simple Gradio demo UI

---

##  References

- Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Kudo & Richardson (2018) — [SentencePiece](https://arxiv.org/abs/1808.06226)
- AI4Bharat — [IndicCorp](https://ai4bharat.iitm.ac.in/)
- OPUS — [OPUS-100 Corpus](https://opus.nlpl.eu/opus-100.php)
- FLORES-200 — [Evaluation Benchmark](https://github.com/facebookresearch/flores)

---

##  Author

**Rajat Malik**
AI/ML Engineer · Chandigarh University · EDSHODH LLP

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/rajat-malik-a62876278)
[![GitHub](https://img.shields.io/badge/GitHub-Rajat25022005-black?style=flat-square&logo=github)](https://github.com/Rajat25022005)

---

##  License

MIT License — see [LICENSE](LICENSE) for details.
