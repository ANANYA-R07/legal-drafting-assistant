# Legal Drafting Assistant - Internship Demonstration Guide

This guide provides a structured, step-by-step walkthrough for demonstrating the **Legal Drafting Assistant** during your internship evaluation or project presentation.

---

## 1. Quick Presentation Outline (5 Minutes)

| Time | Topic | Key Talking Points |
| :--- | :--- | :--- |
| **0:00 - 1:00** | **Project Overview** | Problem in legal drafting, objectives, architecture combining N-gram generation + fine-tuned GPT-2 classification. |
| **1:00 - 2:00** | **Corpus & N-gram Model** | Milestone 1 (data preparation, validation, 5 clause categories) & Milestone 2 (N-gram built from scratch, conditional probabilities, backoff). |
| **2:00 - 3:00** | **GPT-2 Fine-Tuning** | Milestone 3 (real fine-tuning workflow, PyTorch AdamW on CPU, saved weights, sequence classification head). |
| **3:00 - 4:00** | **Live CLI Demonstration** | Milestone 4 (Options 1, 3, 4: generation, manual input, auto-classification pipeline). |
| **4:00 - 5:00** | **Evaluation & Tests** | Milestone 5 (accuracy, precision, recall, confusion matrix on held-out test data) and 35 automated tests passing. |

---

## 2. Environment Preparation

Before starting your presentation, ensure your virtual environment is active:

```powershell
cd C:\Users\Vinayaka\Downloads\legal-drafting-assistant\legal-drafting-assistant
.\.venv\Scripts\Activate.ps1
```

---

## 3. Step-by-Step Live Commands

### Step A: Verify Automated Test Suite (Confidence Check)
Demonstrate that the entire codebase is verified and robust:
```powershell
python -m pytest -v
```
**What to point out:**
- 35 automated tests execute and pass 100%.
- Tests cover data preprocessing, tokenization, N-gram math, GPT-2 inference, evaluation metrics, and CLI subcommands.

---

### Step B: Milestone 1 & 2 - Corpus & N-Gram Generation
Run the standalone generation command to show N-gram capabilities:
```powershell
python -m app.cli generate --seed "The Seller" --n 2 --max-words 18
```
**What to point out:**
- The N-gram model was implemented from scratch (not calling OpenAI/external APIs).
- Explains bigram probability $P(w_i \mid w_{i-1})$.
- Smoothly generates valid legal syntax using the banking & finance corpus.

---

### Step C: Milestone 3 - GPT-2 Clause Classification
Classify a sample confidentiality clause:
```powershell
python -m app.cli classify "The Receiving Party shall not disclose Confidential Information to any third party."
```
Classify a sample payment clause:
```powershell
python -m app.cli classify "Payment is due within thirty days of receipt of invoice."
```
**What to point out:**
- Uses the locally fine-tuned GPT-2 sequence classifier (`models/gpt2_legal_classifier`).
- Outputs predicted category alongside confidence scores.
- Runs 100% locally on CPU without needing external cloud API keys.

---

### Step D: Milestone 4 - Interactive Menu (The "Showstopper")
Launch the main application interface:
```powershell
python -m app.cli
```
**Live actions during presentation:**
1. Type `5` to show **Available clause categories** (shows Confidentiality, Payment, Termination, Liability, Indemnification with sample text).
2. Type `4` to run **Generate & Automatically Classify**.
   - Input seed: `The Buyer`
   - Observe how the N-gram drafts the sentence, and the fine-tuned GPT-2 classifier immediately identifies its clause category with probability breakdown bars!
3. Type `6` to cleanly exit.

---

### Step E: Milestone 5 - Real Evaluation Metrics
Run the evaluation command:
```powershell
python -m app.cli evaluate
```
**What to point out:**
- Metrics are calculated on 18 held-out test clauses (stratified split).
- Shows real Precision, Recall, F1-scores, and Confusion Matrix.
- No fabricated or hardcoded results.

---

## 4. Frequently Asked Questions by Evaluators

**Q1: Did you actually fine-tune GPT-2 or just use prompt engineering?**
> *"We actually fine-tuned a pre-trained `gpt2` checkpoint using PyTorch and Hugging Face `GPT2ForSequenceClassification`. We configured a 5-class sequence classification head, trained with `AdamW` optimizer and cross-entropy loss across 3 epochs on the CPU, and saved the trained weights to `models/gpt2_legal_classifier`."*

**Q2: How does your N-gram model handle words not in the training corpus?**
> *"We implemented Lidstone additive smoothing ($\alpha = 0.01$) and a backoff mechanism. When an input word or bigram context has zero occurrences, the model backs off to the unigram frequency distribution rather than crashing or returning zero probabilities."*

**Q3: How was the dataset split to prevent bias?**
> *"We implemented a stratified train-test split function in `app/corpus.py` with an 80/20 ratio and fixed random seed (42). This ensures every clause category is proportionally represented in both training and evaluation sets."*

---

## 5. Summary of Achievements

- [x] **Milestone 1:** Prepared 100-sentence structured legal corpus with cleaning, validation, and stratified splits.
- [x] **Milestone 2:** Built Unigram, Bigram, and Trigram language models from scratch with smoothing and text generation.
- [x] **Milestone 3:** Fine-tuned GPT-2 sequence classifier on CPU and saved persistent model weights.
- [x] **Milestone 4:** Created beginner-friendly interactive CLI menu and flexible subcommands.
- [x] **Milestone 5:** Evaluated classifier performance with full classification report and confusion matrix.
- [x] **Testing:** 35 passing automated tests with 100% success rate.
