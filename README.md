# Clinical NER on E3C

Multilingual Named Entity Recognition on clinical case reports, using fine-tuned BERT-family models.

The goal is to extract clinical concepts (diseases, syndromes and symptoms, labelled `CLINENTITY`) from unstructured medical text. The task is framed as token classification with BIO tagging.

Course project for **Natural Language Processing**, MSc in Computer Science, University of Catania.

## Dataset

[E3C (European Clinical Case Corpus)](https://github.com/hltfbk/E3C-Corpus), Layer 1 (gold standard, manually annotated by domain experts). It contains clinical case reports in five languages: Italian, English, French, Spanish and Basque.

The original annotations are in UIMA XMI format (WebAnno). `generate_dataset.py` does three things:

- converts them to JSON, keeping the raw text and the entity offsets;
- creates an 80/20 train/validation split;
- builds training pools of increasing size (1, 2, 5, 10, 20, 50, 100, 150, 200 documents and full).

## Experiments

| Experiment | Model | Training data | Best F1 |
| --- | --- | --- | --- |
| Few-shot scaling | mBERT | 1 document | 12.48% |
| Few-shot scaling | mBERT | 50 documents | 64.28% |
| Few-shot scaling | mBERT | 200 documents | 72.55% |
| Multilingual fine-tuning | mBERT | full, all languages | 75.70% |
| Zero-shot cross-lingual transfer | BERT Italian cased | full, Italian only | 51.37% |
| Multilingual fine-tuning | XLM-RoBERTa large | full, all languages | **78.77%** |

The F1 score in the table is the Exact/Partial Match score described below.

**Evaluation metric.** Two strategies were compared:

- **Exact match**, using `seqeval`.
- **Exact/Partial match** (SemEval style). An overlapping prediction with inexact boundaries counts as 0.5 true positive.

The partial variant scores about 10 points higher. It also better reflects model quality, because the boundaries of multi-word clinical entities are often ambiguous even in the gold standard.

**Findings.**

- Performance grows almost linearly between 5 and 50 documents.
- It saturates after roughly 150 to 200 documents.
- Multilingual fine-tuning outperforms zero-shot transfer from Italian by about 24 points.
- XLM-RoBERTa gives the best overall result.

All runs are tracked with Weights & Biases. The plots are in `charts/`.

## Repository structure

```
.
├── scripts/
│   ├── generate_dataset.py   # XMI parsing, train/val split, few-shot pools
│   └── train_model.py        # BIO encoding, fine-tuning, evaluation metrics
├── experiments.ipynb         # preprocessing walkthrough, result analysis, inference demo
├── charts/                   # result plots
├── example/                  # sample clinical reports (EN, FR) for inference
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

1. Download the E3C corpus into `data/raw/E3C-Corpus-2.0.0/`.
2. Create a `.env` file containing `WANDB_API_KEY=<your key>`.
3. Generate the datasets and train:

```bash
python scripts/generate_dataset.py
python scripts/train_model.py
```

The model, the training file and the run name are set at the top of `train_model.py`.

Training settings are:

- learning rate 2e-5;
- batch size 8;
- linear scheduler;
- bf16;
- best checkpoint selected on validation loss.

The last section of `experiments.ipynb` is an interactive demo. It runs a fine-tuned model on unseen reports and returns the clinical entities found with confidence of at least 0.8.

## Author

Edoardo Tantari
