"""Unit tests for the evaluation module."""

import pytest
from app.corpus import LEGAL_CLAUSE_CATEGORIES
from app.evaluation import evaluate_model


def test_evaluate_model_structure():
    test_data = [
        {"clause_type": "Confidentiality", "sentence": "The Receiving Party shall keep information confidential."},
        {"clause_type": "Payment", "sentence": "The Buyer shall pay invoice within thirty days."},
        {"clause_type": "Termination", "sentence": "Either party may terminate upon thirty days written notice."},
        {"clause_type": "Liability", "sentence": "Neither party shall be liable for indirect damages."},
        {"clause_type": "Indemnification", "sentence": "The Seller shall indemnify the Buyer against all claims."},
    ]
    results = evaluate_model(test_data=test_data, save_results=False)

    assert isinstance(results, dict)
    assert results["total_test_samples"] == 5
    assert 0.0 <= results["accuracy"] <= 1.0
    assert 0.0 <= results["precision_macro"] <= 1.0
    assert 0.0 <= results["recall_macro"] <= 1.0
    assert 0.0 <= results["f1_macro"] <= 1.0
    assert len(results["confusion_matrix"]) == len(LEGAL_CLAUSE_CATEGORIES)
    assert "classification_report_text" in results


def test_evaluate_model_empty_test_data():
    with pytest.raises(ValueError):
        evaluate_model(test_data=[])
