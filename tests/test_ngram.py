"""Unit tests for N-gram language models (Unigram, Bigram, Trigram)."""

import pytest
from app.ngram import NgramModel


@pytest.fixture
def sample_sentences():
    return [
        "The Receiving Party shall hold all Confidential Information in strict confidence.",
        "The Receiving Party shall not disclose Confidential Information to any third party.",
        "The Buyer shall pay the Purchase Price in accordance with the payment schedule.",
        "Either party may terminate this Agreement upon thirty days written notice.",
        "Neither party shall be liable for indirect or consequential damages.",
    ]


def test_unigram_model(sample_sentences):
    model = NgramModel(n=1)
    model.train(sample_sentences)
    assert model.is_trained is True
    assert model.total_tokens > 0

    prob = model.get_unigram_prob("party")
    assert prob > 0.0

    preds = model.predict_next("", top_k=3)
    assert len(preds) > 0


def test_bigram_model(sample_sentences):
    model = NgramModel(n=2)
    model.train(sample_sentences)

    prob = model.get_bigram_prob("party", "receiving")
    assert prob > 0.0

    preds = model.predict_next("receiving", top_k=3)
    words = [w for w, _ in preds]
    assert "party" in words


def test_trigram_model(sample_sentences):
    model = NgramModel(n=3)
    model.train(sample_sentences)

    prob = model.get_trigram_prob("shall", "receiving", "party")
    assert prob > 0.0

    preds = model.predict_next("the receiving party", top_k=3)
    words = [w for w, _ in preds]
    assert "shall" in words


def test_ngram_empty_input_handling(sample_sentences):
    model = NgramModel(n=2)
    model.train(sample_sentences)

    preds = model.predict_next("", top_k=5)
    assert len(preds) > 0

    generated = model.generate_sentence(seed="", max_words=10)
    assert isinstance(generated, str)
    assert len(generated) > 0


def test_ngram_unknown_words_handling(sample_sentences):
    model = NgramModel(n=2)
    model.train(sample_sentences)

    # Word not in training vocabulary
    preds = model.predict_next("completely_unknown_token_xyz", top_k=3)
    assert len(preds) > 0
    words, probs = zip(*preds)
    assert all(p >= 0.0 for p in probs)


def test_ngram_sentence_generation_bounds(sample_sentences):
    model = NgramModel(n=2)
    model.train(sample_sentences)

    generated = model.generate_sentence(seed="The Buyer", max_words=12)
    assert len(generated.split()) <= 15
    assert generated.endswith(".")


def test_ngram_generate_document(sample_sentences):
    model = NgramModel(n=2)
    model.train(sample_sentences)

    doc = model.generate_document(sentences_per_category=1)
    assert isinstance(doc, dict)
    assert "Confidentiality" in doc
    assert len(doc["Confidentiality"]) == 1

    formatted = model.format_legal_document(doc)
    assert "COMMERCIAL AGREEMENT" in formatted
    assert "SECTION 1:" in formatted


def test_invalid_n_order():
    with pytest.raises(ValueError):
        NgramModel(n=5)


def test_train_with_empty_dataset():
    model = NgramModel(n=2)
    with pytest.raises(ValueError):
        model.train([])


def test_predict_before_train():
    model = NgramModel(n=2)
    with pytest.raises(RuntimeError):
        model.predict_next("The")
