# Legal Drafting Assistant

[![Python 3.11](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15.0%2Bcpu-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![Tests](https://img.shields.io/badge/Tests-38%20Passed-brightgreen.svg)]()

A comprehensive AI-powered legal drafting and clause classification assistant built for paralegals and legal professionals in the Banking & Finance domain. The tool generates contract clauses using an **N-gram next-word prediction model built from scratch** and automatically classifies clauses into standard legal categories using a **fine-tuned GPT-2 sequence classifier**.

---

## Table of Contents

1. [Project Introduction](#1-project-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [Technologies Used](#4-technologies-used)
5. [Project Architecture](#5-project-architecture)
6. [Folder Structure](#6-folder-structure)
7. [Installation Instructions for Windows](#7-installation-instructions-for-windows)
8. [Virtual Environment Setup](#8-virtual-environment-setup)
9. [Dependency Installation](#9-dependency-installation)
10. [Dataset Explanation](#10-dataset-explanation)
11. [N-Gram Language Model Implementation](#11-n-gram-language-model-implementation)
12. [GPT-2 Fine-Tuning Process](#12-gpt-2-fine-tuning-process)
13. [How to Run the Application](#13-how-to-run-the-application)
14. [Sample Inputs and Outputs](#14-sample-inputs-and-outputs)
15. [Model Evaluation Results](#15-model-evaluation-results)
16. [Limitations and Future Enhancements](#16-limitations-and-future-enhancements)

---

## 1. Project Introduction

Drafting contract agreements requires strict adherence to legal language norms, consistent terminology, and accurate clause categorization. The **Legal Drafting Assistant** is a command-line application that bridges statistical language modeling with modern deep learning.

Paralegals can:
- **Draft standard legal sentences** word-by-word or from starting seed phrases using unigram, bigram, and trigram models.
- **Categorize contract clauses** instantly into 5 essential legal categories: *Confidentiality, Payment, Termination, Liability, and Indemnification*.
- **Run an integrated pipeline** that drafts a clause and verifies its classification in real time.
- **Evaluate classifier metrics** directly on held-out test data.

---

## 2. Problem Statement

Junior paralegals and legal drafters frequently encounter two bottlenecks:
1. **Repetitive drafting friction:** Drafting standard contract covenants and warranties takes considerable time, often resulting in syntactical inconsistencies.
2. **Clause misclassification:** Organizing clauses during contract reviews, due diligence, and negotiation requires manually tagging provisions to verify coverage.

Existing large language model APIs can be expensive, require persistent network connectivity, and expose private contract data to external third-party cloud services. This project implements a lightweight, local, explainable solution combining statistical N-grams and local fine-tuned transformer inference.

---

## 3. Project Objectives

- **Milestone 1:** Create and preprocess a structured legal text corpus covering 5 standard commercial clause categories with rigorous data validation and stratified splitting.
- **Milestone 2:** Implement an N-gram language model from scratch (unigram, bigram, trigram) with frequency calculation, conditional probabilities, Lidstone smoothing, backoff, and text generation.
- **Milestone 3:** Fine-tune a pre-trained GPT-2 model with a sequence classification head on labeled legal clauses, with persistent weight saving and inference.
- **Milestone 4:** Design an interactive, beginner-friendly Command-Line Interface (CLI) combining generation and classification workflows.
- **Milestone 5:** Evaluate the classification model on held-out test clauses, generating accuracy, precision, recall, F1-scores, confusion matrices, and classification reports without hardcoded values.

---

## 4. Technologies Used

- **Language:** Python 3.11.9
- **Deep Learning Framework:** PyTorch 2.15.0+cpu
- **NLP / Transformers:** Hugging Face `transformers` 5.23.0 (`GPT2ForSequenceClassification`, `AutoTokenizer`), `tokenizers`, `datasets`
- **Evaluation & Metrics:** `scikit-learn` 1.9.1, `numpy` 2.4.6, `pandas` 3.1.1
- **Testing:** `pytest` 9.2.0
- **HTTP / Utilities:** `httpx` 0.28.1

---

## 5. Project Architecture

```
+-------------------------------------------------------------+
|                Command-Line Interface (CLI)                 |
|                   (app/cli.py - Menu & Args)                |
+------------------------------+------------------------------+
                               |
       +-----------------------+-----------------------+
       |                                               |
+------v-----------------------+       +---------------v---------------+
|        Milestone 2           |       |        Milestones 3 & 5       |
|    N-gram Language Model     |       |    GPT-2 Clause Classifier    |
|       (app/ngram.py)         |       |   (app/gpt2_classifier.py)    |
|                              |       |                               |
|  * Unigram, Bigram, Trigram  |       |  * Sequence Classification    |
|  * Conditional Probabilities |       |  * 5 Clause Categories        |
|  * Backoff & Smoothing       |       |  * Softmax Confidence Scores  |
|  * Next-Word Prediction      |       |  * Evaluation (app/eval.py)   |
|  * Legal Sentence Generator  |       |  * Confusion Matrix & F1      |
+--------------^---------------+       +---------------^---------------+
               |                                       |
+--------------+---------------------------------------+---------------+
|                         Milestone 1                                  |
|                 Legal Text Corpus & Pipeline                         |
|                       (app/corpus.py)                                |
|                                                                      |
|  * Raw Corpus (data/legal_corpus.csv)                                |
|  * Cleaning, Normalization & Word Tokenization                       |
|  * Stratified Train/Test Split (80% Train / 20% Test)                |
|  * Structured Output (data/processed_corpus.json)                    |
+----------------------------------------------------------------------+
```

---

## 6. Folder Structure

```
legal-drafting-assistant/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Continuous Integration GitHub Actions
├── .vscode/
│   └── settings.json              # VS Code Python environment & path configuration
├── app/
│   ├── __init__.py                # Package initialization
│   ├── cli.py                     # Interactive CLI & subcommand handlers
│   ├── core.py                    # Milestone entry points (milestone_1 through 5)
│   ├── corpus.py                  # Milestone 1: Corpus loader, cleaner, tokenizer, splitter
│   ├── evaluation.py              # Milestone 5: Accuracy, F1, confusion matrix evaluation
│   ├── gpt2_classifier.py         # Milestone 3: GPT-2 fine-tuning & inference engine
│   ├── llm.py                     # Optional external LLM chat adapter (Ollama/Groq/Gemini)
│   ├── ngram.py                   # Milestone 2: Unigram, Bigram, Trigram models from scratch
│   └── storage.py                 # Key-value JSON storage utility
├── data/
│   ├── evaluation_report.txt      # Formatted evaluation report text
│   ├── evaluation_results.json    # Numerical evaluation metrics
│   ├── legal_corpus.csv           # Master legal clause dataset (100 sentences)
│   └── processed_corpus.json      # Structured, tokenized, and split dataset
├── models/
│   └── gpt2_legal_classifier/     # Saved fine-tuned GPT-2 weights and tokenizer
│       ├── classifier_metadata.json
│       ├── config.json
│       ├── merges.txt
│       ├── model.safetensors
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── vocab.json
├── tests/
│   ├── test_cli.py                # Tests for all CLI commands and menus
│   ├── test_core.py               # Tests for all 5 milestones
│   ├── test_corpus.py             # Tests for corpus loading, cleaning, validation
│   ├── test_evaluation.py         # Tests for metrics and confusion matrix
│   ├── test_ngram.py              # Tests for unigram, bigram, trigram, backoff
│   └── test_smoke.py              # Smoke tests for plumbing
├── DEMO_GUIDE.md                  # Step-by-step presentation demonstration guide
├── pytest.ini                     # Pytest configuration
├── README.md                      # Comprehensive project documentation
└── requirements.txt               # Project dependencies
```

---

## 7. Installation Instructions for Windows

### Prerequisites
- Windows 10 or 11 (64-bit)
- Python 3.11.9
- Visual Studio Code or PowerShell terminal
- Git

### Step 1: Clone or Navigate to the Project
```powershell
cd C:\Users\Vinayaka\Downloads\legal-drafting-assistant\legal-drafting-assistant
```

---

## 8. Virtual Environment Setup

Verify Python 3.11 is installed, then create and activate the virtual environment:

```powershell
# Using Python 3.11 launcher
py -3.11 -m venv .venv

# Activate virtual environment in PowerShell
.\.venv\Scripts\Activate.ps1
```

*(If PowerShell script execution is restricted, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

---

## 9. Dependency Installation

Install all required packages into `.venv`:

```powershell
# Install PyTorch CPU build (fast and lightweight)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

Verify installation:
```powershell
python -c "import torch, transformers, sklearn; print('Ready! PyTorch:', torch.__version__)"
```

---

## 10. Dataset Explanation

The corpus is located at `data/legal_corpus.csv` and contains curated, realistic legal sentences across 5 commercial categories:

1. **Confidentiality:** Non-disclosure covenants, trade secret protections, and permitted disclosure boundaries.
2. **Payment:** Purchase prices, payment schedules, late fees, invoicing timelines, and taxes.
3. **Termination:** Notice periods, immediate termination for material breach, insolvency, and transition covenants.
4. **Liability:** Indirect/consequential damages exclusions, cumulative liability caps, and warranty disclaimers.
5. **Indemnification:** Third-party defense obligations, hold-harmless provisions, and legal fee reimbursement.

### Preprocessing and Validation Pipeline:
- **Normalization:** Cleans whitespace, removes non-printable characters, normalizes typographic quotes (`“”` $\rightarrow$ `""`) and em-dashes.
- **Validation:** Ensures non-empty sentences, minimum word length ($\ge 3$ words), and validates clause category labels.
- **Stratified Split:** Partitions the corpus into 80% training (79 samples) and 20% testing (18 samples), guaranteeing balanced representation across all 5 classes.
- **Storage:** Persisted to `data/processed_corpus.json` with metadata statistics.

---

## 11. N-Gram Language Model Implementation

The N-gram model (`app/ngram.py`) is written **entirely from scratch** without external generation libraries:

- **Unigram ($n=1$):** Models unconditional word probabilities $P(w) = \frac{C(w)}{N}$.
- **Bigram ($n=2$):** Models conditional probability given the preceding word:
  $$P(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$
- **Trigram ($n=3$):** Models conditional probability given the preceding two words:
  $$P(w_i \mid w_{i-2}, w_{i-1}) = \frac{C(w_{i-2}, w_{i-1}, w_i)}{C(w_{i-2}, w_{i-1})}$$
- **Smoothing & Backoff:** Uses Lidstone/Laplace smoothing ($\alpha = 0.01$) to prevent zero-probability errors. If a trigram context is unseen, it gracefully backs off to the bigram; if the bigram context is unseen, it backs off to smoothed unigrams.
- **Text Generation:** Supports temperature-scaled sampling ($T > 0$) and greedy argmax ($T = 0$), respecting sentence boundaries with `<s>` and `</s>` tokens.

---

## 12. GPT-2 Fine-Tuning Process

### Architecture
- **Base Checkpoint:** `gpt2` (124M parameters, 12 layers, 12 attention heads).
- **Classification Head:** Linear projection layer mapping GPT-2's pooled hidden states to 5 output logits representing the clause categories.
- **Padding Configuration:** GPT-2 has no native pad token; `tokenizer.pad_token = tokenizer.eos_token` and `model.config.pad_token_id = tokenizer.eos_token_id` are explicitly set.

### Training Configuration
- **Device:** CPU (practical and accessible for standard development machines).
- **Epochs:** 3
- **Batch Size:** 4
- **Optimizer:** `AdamW` with learning rate $5 \times 10^{-5}$ and weight decay $0.01$.
- **Gradient Clipping:** Max norm $1.0$.
- **Training Loss Progression:**
  - Epoch 1: Average Loss = 2.7850
  - Epoch 2: Average Loss = 1.6970
  - Epoch 3: Average Loss = 1.1943
- **Saved Model:** Weights and tokenizer are saved to `models/gpt2_legal_classifier/` (`model.safetensors`, `config.json`, `vocab.json`).

---

## 13. How to Run the Application

### 1. Interactive Main Menu
Launch the interactive terminal UI:
```powershell
python -m app.cli
```
or explicitly:
```powershell
python -m app.cli menu
```

### 2. Direct Command-Line Subcommands
```powershell
# Check plumbing
python -m app.cli hello

# Generate a legal sentence with bigram model
python -m app.cli generate --seed "The Seller" --n 2 --max-words 20

# Classify a legal sentence
python -m app.cli classify "The Receiving Party shall not disclose Confidential Information to any third party."

# Generate and automatically classify
python -m app.cli pipeline --seed "The Buyer"

# View all 5 categories
python -m app.cli categories

# Run classification evaluation
python -m app.cli evaluate

# Generate a complete multi-clause commercial agreement draft
python -m app.cli document

# View recent drafting and classification history
python -m app.cli history
```

---

## 14. Sample Inputs and Outputs

### Sample 1: Interactive Pipeline (Generate + Auto-Classify)
```
[Generated Legal Sentence]:
  "The Buyer shall pay the purchase price in accordance with the payment schedule."

[Automatic Classification]:
  Predicted Category : Payment
  Confidence Score   : 88.42%

[Category Probabilities]:
   • Confidentiality    :   2.10% | 
   • Payment            :  88.42% | ████████████████████
   • Termination        :   3.25% | 
   • Liability          :   4.11% | 
   • Indemnification    :   2.12% | 
```

### Sample 2: Next-Word Prediction
```
Enter seed phrase: The Receiving Party
[Next-word predictions following 'Receiving Party']:
   • shall           (54.12%)
   • may             (18.25%)
   • agrees          (12.40%)
   • assumes         (8.10%)
   • warrants        (7.13%)
```

---

## 15. Model Evaluation Results

Evaluation was executed on **18 held-out test clauses** from the stratified test split:

```
=================================================================
LEGAL CLAUSE CLASSIFICATION EVALUATION REPORT
Model: Fine-Tuned GPT-2 Sequence Classifier
=================================================================

Total Test Samples: 18
Overall Accuracy:   38.89%
Macro Precision:    51.27%
Macro Recall:       36.67%
Macro F1-Score:     33.11%

CLASSIFICATION REPORT:
                 precision    recall  f1-score   support

Confidentiality       1.00      0.33      0.50         3
        Payment       1.00      0.25      0.40         4
    Termination       0.00      0.00      0.00         3
      Liability       0.36      1.00      0.53         4
Indemnification       0.20      0.25      0.22         4

       accuracy                           0.39        18
      macro avg       0.51      0.37      0.33        18
   weighted avg       0.51      0.39      0.34        18

CONFUSION MATRIX:
True \ Pred       Confiden   Payment  Terminat   Liabilit  Indemnif
-----------------------------------------------------------------
Confidentiality          1         0         0          1         1
Payment                  0         1         0          2         1
Termination              0         0         0          1         2
Liability                0         0         0          4         0
Indemnification          0         0         0          3         1
=================================================================
```

---

## 16. Limitations and Future Enhancements

### Current Limitations:
1. **Dataset Size:** 100 sentences is suitable for educational demonstration and CPU training, but real-world legal models benefit from thousands of clauses.
2. **CPU Training Epochs:** Trained for 3 epochs with a small batch size to ensure quick training (~2 minutes) on standard CPU hardware. Further training epochs with GPU acceleration will increase multi-class recall.
3. **Domain Vocabulary:** Focuses on Banking & Finance commercial contracts. Specialized areas (e.g. Maritime law, IP patent licensing) require domain-specific expansions.

### Future Enhancements:
- **LoRA / PEFT Integration:** Utilize parameter-efficient fine-tuning (Low-Rank Adaptation) for faster fine-tuning and reduced checkpoint sizes.
- **Web UI:** Wrap the CLI in a FastAPI backend with a Streamlit or React frontend for interactive drag-and-drop document tagging.
- **Contract Risk Scoring:** Add clause risk assessment alongside classification to flag unfavorable liability caps.

---

## Running Automated Tests

Run the full pytest suite with verbose output:

```powershell
python -m pytest -v
```

**Result: 38 passed in ~29 seconds (100% test passing rate).**
