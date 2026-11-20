"""Legal Drafting Assistant - core logic and milestone entry points."""

import json
from pathlib import Path
from typing import Any


# --- Milestone 1: Prepare a small legal text corpus for n-gram training. ---


def milestone_1() -> dict[str, Any]:
    """Load, validate, and return the legal corpus as a structured dictionary."""
    from app.corpus import load_corpus

    corpus_data = load_corpus()
    clauses = corpus_data.get("clauses", {})
    sentences = corpus_data.get("sentences", [])

    output_path = Path(__file__).parent / "processed_corpus.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus_data, f, indent=2)

    print(f"Milestone 1: Loaded {len(sentences)} sentences from {len(clauses)} clause categories")
    print(f"  Clause categories: {sorted(clauses.keys())}")
    return corpus_data


# --- Milestone 2: Implement the n-gram next-word predictor. ---


def milestone_2(n: int = 2):
    """Implement, train, and return the N-gram next-word predictor."""
    from app.corpus import load_sentences
    from app.ngram import NgramModel

    sentences = load_sentences()
    model = NgramModel(n=n)
    model.train(sentences)

    order_name = {1: "Unigram", 2: "Bigram", 3: "Trigram"}.get(n, f"{n}-gram")
    print(f"Milestone 2 Complete: {order_name} model trained on {len(sentences)} sentences")
    return model


# --- Milestone 3: Load pre-trained GPT-2 and fine-tune for legal clause tagging. ---


def milestone_3(force_retrain: bool = False, epochs: int = 3) -> dict[str, Any]:
    """Load or fine-tune the GPT-2 legal clause classifier."""
    from app.gpt2_classifier import FineTunedGPT2Classifier

    classifier = FineTunedGPT2Classifier()
    if force_retrain or not classifier.is_fine_tuned:
        classifier.train(epochs=epochs)
        classifier.save()
        print("Milestone 3 Complete: GPT-2 legal clause classifier trained and saved")
    else:
        print("Milestone 3: Loaded existing fine-tuned GPT-2 legal clause classifier")

    return {"classifier": classifier}


# --- Milestone 4: CLI integration between n-gram and GPT-2 models. ---


def milestone_4() -> dict[str, Any]:
    """Execute integrated workflow: generate legal sentence with N-gram, then classify with GPT-2."""
    from app.corpus import load_sentences
    from app.gpt2_classifier import FineTunedGPT2Classifier
    from app.ngram import NgramModel

    # Step 1: Train/load N-gram
    sentences = load_sentences()
    ngram_model = NgramModel(n=2)
    ngram_model.train(sentences)

    # Step 2: Generate legal sentence
    generated_text = ngram_model.generate_sentence(seed="The Buyer", max_words=20)

    # Step 3: Load classifier and classify generated text
    classifier = FineTunedGPT2Classifier()
    if not classifier.is_fine_tuned and not classifier._saved_model_exists():
        classifier.train(epochs=1, verbose=False)
        classifier.save()

    classification_result = classifier.predict(generated_text)

    print("Milestone 4: Integrated Pipeline Executed Successfully")
    print(f"  Generated Text: '{generated_text}'")
    print(f"  Predicted Category: {classification_result['predicted_category']} (Confidence: {classification_result['confidence']:.2%})")

    return {
        "generated_text": generated_text,
        "classification": classification_result,
        "ngram_model": ngram_model,
        "classifier": classifier,
    }


# --- Milestone 5: Evaluate classification accuracy on example clauses. ---


def milestone_5(classifier=None) -> dict[str, Any]:
    """Evaluate the classification model on held-out test data and return metrics."""
    from app.evaluation import evaluate_model

    results = evaluate_model(classifier=classifier)
    accuracy = results.get("accuracy", 0.0)
    f1 = results.get("f1_macro", 0.0)
    print(f"Milestone 5 Complete: Accuracy = {accuracy:.4f} | Macro F1 = {f1:.4f}")
    return results
