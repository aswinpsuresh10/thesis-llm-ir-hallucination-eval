# Evaluating Hallucination and Trust in LLM-Enhanced Information Retrieval Systems

> Master Thesis | Universität Koblenz | 2026  
> Author: Aswin Panthithara Suresh  

---

## Overview

This thesis evaluates hallucination rates, semantic drift, and user trust in LLM-augmented information retrieval systems. It benchmarks **GPT-J (6B)**, **Gemini 2.0 Flash**, and **Gemini 2.5 Flash** across three fact-checking datasets using a multi-stage evaluation pipeline covering parametric recall, veracity classification, claim synthesis, and LLM-as-a-Judge auditing.

---

## Repository Structure

```
llm-ir-hallucination-eval/
│
├── src/
│   ├── config.py                        # Central config: paths, models, hyperparameters
│   ├── models/
│   │   ├── base.py                      # Abstract LLM interface
│   │   ├── gptj.py                      # GPT-J (6B) via HuggingFace
│   │   └── gemini.py                    # Gemini 2.0 Flash + 2.5 Flash
│   ├── datasets/
│   │   ├── simpleqa.py                  # SimpleQA loader
│   │   ├── fever.py                     # FEVER loader
│   │   └── scifact.py                   # SciFact loader
│   ├── pipelines/
│   │   ├── parametric_recall.py         
│   │   ├── veracity_classification.py   
│   │   ├── claim_synthesis.py           
│   │   └── llm_as_judge.py              
│   ├── evaluation/
│   │   ├── scoring.py                   # Accuracy, hallucination & generative metrics
│   │   ├── metrics.py                   # Semantic similarity metrics
│   │   └── bias_detection.py            # WEAT + SEAT from scratch
│   └── utils/
│       ├── prompts.py                   # All zero-shot prompt templates
│       ├── logger.py                    # Rotating file + console logger
│       └── io.py                        # Save/load results (CSV + JSON)
│
├── data/                                # Raw & processed datasets (gitignored)
├── results/                             # Pipeline outputs (gitignored)
├── run_exp.py                   
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Datasets

| Dataset | Task | Labels |
|---|---|---|
| [SimpleQA](https://github.com/openai/simple-evals) | Short-form QA — Parametric Recall | Correct / Incorrect / Not Attempted |
| [FEVER](https://fever.ai) | Fact Verification | SUPPORTS / REFUTES / NOT ENOUGH INFO |
| [SciFact](https://github.com/allenai/scifact) | Scientific Claim Verification | SUPPORTS / REFUTES |

> Raw data is gitignored. Datasets are downloaded automatically from HuggingFace on first run and cached to `data/raw/`.

---

## Evaluation Framework

| Stage | Task | Dataset | Metric |
|---|---|---|---|
| 1 — Parametric Recall | Zero-shot short-form QA | SimpleQA | Overall Accuracy, Hallucination Rate, Abstention Rate |
| 2a — Veracity Classification | Binary True/False (no evidence) | FEVER, SciFact | Acc_veracity (Strict Match) |
| 2b — Claim Synthesis | Evidence-grounded claim generation | FEVER, SciFact | BLEU, ROUGE-L, BERTScore F1 |
| 2c — LLM-as-a-Judge | Auditor reviews generated claims | FEVER, SciFact | Generative Accuracy (Acc_gen) |

**Bias Detection** — WEAT and SEAT applied to model outputs to measure implicit stereotype reinforcement.

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/your-username/llm-ir-hallucination-eval.git
cd llm-ir-hallucination-eval

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY and HUGGINGFACE_TOKEN
```

### Running the Pipeline

```bash
python run_exp.py              # Full run (all models, all stages)
python run_exp.py --dev        # Dev run (100 samples per dataset)
python run_exp.py --stage 1    # Stage 1 only (Parametric Recall)
python run_exp.py --model gemini  # Gemini models only
```

---

## License

This repository is for academic submission purposes. All rights reserved unless otherwise stated.