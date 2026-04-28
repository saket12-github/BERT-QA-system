# BERT QA System

A lightweight Question Answering (QA) application built with Hugging Face Transformers and Gradio.
It uses a RoBERTa model fine-tuned on SQuAD 2.0 to extract answer spans from user-provided context paragraphs.

This repository is based on the project work described in `Research_Paper.pdf`:
**"Encoder-based LLMs: Building QA Systems and Comparative Analysis"**.

## Overview

This project provides:

- A web UI to ask natural-language questions over a context paragraph.
- A reusable QA engine wrapper with input validation and inference metadata.
- A script to evaluate model quality (Exact Match and F1) on SQuAD 2.0.
- Research context and comparative insights across BERT-family encoder models.

The default model is:

- `IProject-10/roberta-base-finetuned-squad2`

## Features

- Span-based extractive QA (`question-answering` pipeline)
- Basic no-answer handling with a confidence threshold
- Inference metadata: confidence, character span, latency, and no-answer flag
- Batch evaluation on SQuAD 2.0 with JSON report output
- Simple Gradio interface for interactive testing
- Paper-aligned framing for encoder-model QA benchmarking

## Research Context (From Paper)

The accompanying paper studies encoder-based transformer models for extractive QA and compares:

- BERT
- RoBERTa
- DistilBERT
- ALBERT
- XLM-RoBERTa

### Task and Dataset

- Task: Answer Retrieval Question Answering (extractive QA)
- Dataset: SQuAD 2.0
- Data splits reported in the paper:
  - Train: 130,319 samples
  - Validation: 11,873 samples

### Training and Evaluation Setup (Paper)

- Fine-tuning done on Google Colab with NVIDIA Tesla T4 GPU
- Core training hyperparameters reported:
  - Batch size: 16
  - Epochs: 3
  - Learning rate: `3e-5`
  - Weight decay: `0.01`
- Primary evaluation metrics:
  - Exact Match (EM)
  - F1 score

### Comparative Results (Reported)

| Model | EM | F1 |
|---|---:|---:|
| BERT | 73.50 | 76.79 |
| RoBERTa | **79.72** | **83.05** |
| DistilBERT | 65.88 | 68.98 |
| ALBERT | 78.13 | 81.55 |
| XLM-RoBERTa | 75.52 | 78.73 |

Key takeaways from the paper:

- RoBERTa achieved the best EM/F1 in the reported experiments.
- DistilBERT offered faster fine-tuning with lower accuracy.
- ALBERT provided a strong performance/efficiency trade-off.
- XLM-RoBERTa delivered competitive scores with multilingual potential.

## Tech Stack

- Python 3.11
- [Transformers](https://github.com/huggingface/transformers)
- [Datasets](https://github.com/huggingface/datasets)
- [PyTorch](https://pytorch.org/)
- [Gradio](https://www.gradio.app/)

## Project Structure

```text
BERT-QA-system/
├── app.py            # Gradio app entrypoint
├── qa_engine.py      # QA pipeline wrapper + validation
├── evaluate.py       # Evaluation script for SQuAD 2.0
├── metrics.py        # EM/F1 metric utilities
├── requirements.txt  # Python dependencies
└── runtime.txt       # Runtime version metadata
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/BERT-QA-system.git
cd BERT-QA-system
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Web App

Start the Gradio app:

```bash
python app.py
```

Then open the local Gradio URL shown in the terminal (typically `http://127.0.0.1:7860`).

## Evaluate Model Performance

Run a quick benchmark on the SQuAD 2.0 validation split:

```bash
python evaluate.py --split validation --max-samples 200 --output evaluation_results.json
```

Example output fields:

- `exact_match`
- `f1`
- `avg_latency_ms`
- `throughput_samples_per_sec`

Note: this repository currently evaluates the configured model checkpoint (default: RoBERTa checkpoint). Reproducing the full multi-model comparison from the paper requires running equivalent fine-tuning/evaluation across all listed models.

## Usage Notes

- Keep context length under the engine limit (`MAX_CONTEXT_CHARS = 6000`).
- If the model confidence is below threshold, the app returns:
  - `No confident answer found in context.`
- Evaluation uses best-match scoring across all available ground-truth answers.

## Deployment

This repository is ready for GitHub hosting and can also be adapted for:

- Hugging Face Spaces (Gradio SDK)
- Local demos for NLP coursework/projects
- Lightweight QA API prototyping

## Citation

If you use this repository in academic or project work, cite the associated paper:

- Saket Chaudhari and Shalini Dangi, *Encoder-based LLMs: Building QA systems and Comparative Analysis*.

## License

MIT License