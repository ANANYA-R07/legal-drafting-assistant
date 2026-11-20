"""Milestone 5: Classification Evaluation on held-out legal test data.

Calculates Accuracy, Precision, Recall, F1-score, Confusion Matrix,
and per-category Classification Report using scikit-learn.
"""

import json
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.corpus import LEGAL_CLAUSE_CATEGORIES, get_default_data_dir, load_corpus
from app.gpt2_classifier import FineTunedGPT2Classifier


def evaluate_model(
    classifier: FineTunedGPT2Classifier | None = None,
    test_data: list[dict[str, str]] | None = None,
    save_results: bool = True,
) -> dict[str, Any]:
    """Evaluate GPT-2 classification model on test data and calculate metrics.
    
    Args:
        classifier: FineTunedGPT2Classifier instance or None to instantiate/load saved.
        test_data: Optional test dataset list. If None, loaded from load_corpus()["test"].
        save_results: Whether to save results to data directory.
        
    Returns:
        Structured dictionary with metrics, classification report, and confusion matrix.
    """
    if test_data is None:
        corpus = load_corpus()
        test_data = corpus["test"]

    if not test_data:
        raise ValueError("Test dataset is empty. Cannot perform evaluation.")

    if classifier is None:
        classifier = FineTunedGPT2Classifier()
        if not classifier.is_fine_tuned and not classifier._saved_model_exists():
            # If not yet trained, run training to fine-tune model
            classifier.train()
            classifier.save()

    test_sentences = [item["sentence"] for item in test_data]
    true_labels = [item["clause_type"] for item in test_data]

    # Run predictions on actual test data
    predictions = classifier.predict_batch(test_sentences)
    pred_labels = [p["predicted_category"] for p in predictions]

    # Calculate metrics
    accuracy = float(accuracy_score(true_labels, pred_labels))
    precision_macro = float(precision_score(true_labels, pred_labels, labels=LEGAL_CLAUSE_CATEGORIES, average="macro", zero_division=0))
    recall_macro = float(recall_score(true_labels, pred_labels, labels=LEGAL_CLAUSE_CATEGORIES, average="macro", zero_division=0))
    f1_macro = float(f1_score(true_labels, pred_labels, labels=LEGAL_CLAUSE_CATEGORIES, average="macro", zero_division=0))

    precision_weighted = float(precision_score(true_labels, pred_labels, labels=LEGAL_CLAUSE_CATEGORIES, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(true_labels, pred_labels, labels=LEGAL_CLAUSE_CATEGORIES, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(true_labels, pred_labels, labels=LEGAL_CLAUSE_CATEGORIES, average="weighted", zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=LEGAL_CLAUSE_CATEGORIES).tolist()

    # Classification report
    report_dict = classification_report(
        true_labels,
        pred_labels,
        labels=LEGAL_CLAUSE_CATEGORIES,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        true_labels,
        pred_labels,
        labels=LEGAL_CLAUSE_CATEGORIES,
        zero_division=0,
    )

    results: dict[str, Any] = {
        "total_test_samples": len(test_data),
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "categories": LEGAL_CLAUSE_CATEGORIES,
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "classification_report_text": report_text,
    }

    if save_results:
        data_dir = get_default_data_dir()
        json_path = data_dir / "evaluation_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        report_path = data_dir / "evaluation_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 65 + "\n")
            f.write("LEGAL CLAUSE CLASSIFICATION EVALUATION REPORT\n")
            f.write("Model: Fine-Tuned GPT-2 Sequence Classifier\n")
            f.write("=" * 65 + "\n\n")
            f.write(f"Total Test Samples: {len(test_data)}\n")
            f.write(f"Overall Accuracy:  {accuracy * 100:.2f}%\n")
            f.write(f"Macro Precision:   {precision_macro * 100:.2f}%\n")
            f.write(f"Macro Recall:      {recall_macro * 100:.2f}%\n")
            f.write(f"Macro F1-Score:    {f1_macro * 100:.2f}%\n\n")
            f.write("CLASSIFICATION REPORT:\n")
            f.write(report_text + "\n\n")
            f.write("CONFUSION MATRIX:\n")
            label_header = "True \\ Pred"
            header = f"{label_header:<18}" + "".join(f"{c[:8]:>10}" for c in LEGAL_CLAUSE_CATEGORIES)
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for i, row in enumerate(cm):
                row_str = f"{LEGAL_CLAUSE_CATEGORIES[i]:<18}" + "".join(f"{val:>10}" for val in row)
                f.write(row_str + "\n")
            f.write("=" * 65 + "\n")

    return results
