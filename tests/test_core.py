"""Milestone verification tests for Legal Drafting Assistant."""

from app import core
from app.corpus import LEGAL_CLAUSE_CATEGORIES


def test_milestone_1():
    """Milestone 1: Verify legal text corpus loading, validation, and splitting."""
    corpus = core.milestone_1()
    assert isinstance(corpus, dict)
    assert "clauses" in corpus
    assert "sentences" in corpus
    assert "train" in corpus
    assert "test" in corpus
    assert "categories" in corpus

    # Check categories
    for cat in LEGAL_CLAUSE_CATEGORIES:
        assert cat in corpus["clauses"], f"Category {cat} missing in clauses"
        assert len(corpus["clauses"][cat]) >= 10, f"Expected at least 10 sentences for {cat}"

    # Check non-empty sentences and dataset splits
    assert len(corpus["sentences"]) >= 50
    assert len(corpus["train"]) > len(corpus["test"])
    assert len(corpus["test"]) >= 10


def test_milestone_2():
    """Milestone 2: Verify N-gram language model training, prediction, and generation."""
    model = core.milestone_2(n=2)
    assert model.is_trained is True
    assert model.total_tokens > 0
    assert len(model.vocabulary) > 100

    # Test next-word prediction
    preds = model.predict_next("The Buyer", top_k=5)
    assert isinstance(preds, list)
    assert len(preds) > 0
    words, probs = zip(*preds)
    assert "shall" in words or any(p > 0 for p in probs)

    # Test sentence generation
    generated = model.generate_sentence(seed="The Receiving Party", max_words=15)
    assert isinstance(generated, str)
    assert len(generated.split()) >= 3
    assert generated.endswith(".")


def test_milestone_3():
    """Milestone 3: Verify GPT-2 legal clause classifier loading and inference."""
    result = core.milestone_3(force_retrain=False)
    classifier = result["classifier"]
    assert classifier is not None

    # Test classification on a sample sentence
    sentence = "The Receiving Party shall hold all Confidential Information in strict confidence."
    pred = classifier.predict(sentence)

    assert "predicted_category" in pred
    assert pred["predicted_category"] in LEGAL_CLAUSE_CATEGORIES
    assert "confidence" in pred
    assert 0.0 <= pred["confidence"] <= 1.0
    assert "probabilities" in pred
    assert len(pred["probabilities"]) == len(LEGAL_CLAUSE_CATEGORIES)


def test_milestone_4():
    """Milestone 4: Verify integrated pipeline: N-gram generation followed by GPT-2 classification."""
    pipeline_result = core.milestone_4()
    assert isinstance(pipeline_result, dict)
    assert "generated_text" in pipeline_result
    assert "classification" in pipeline_result

    generated_text = pipeline_result["generated_text"]
    classification = pipeline_result["classification"]

    assert len(generated_text) > 0
    assert classification["predicted_category"] in LEGAL_CLAUSE_CATEGORIES
    assert classification["confidence"] > 0.0


def test_milestone_5():
    """Milestone 5: Verify evaluation metrics calculation on test data."""
    results = core.milestone_5()
    assert isinstance(results, dict)
    assert "accuracy" in results
    assert "f1_macro" in results
    assert "precision_macro" in results
    assert "recall_macro" in results
    assert "confusion_matrix" in results
    assert "classification_report" in results

    assert 0.0 <= results["accuracy"] <= 1.0
    assert 0.0 <= results["f1_macro"] <= 1.0
    assert len(results["confusion_matrix"]) == len(LEGAL_CLAUSE_CATEGORIES)
