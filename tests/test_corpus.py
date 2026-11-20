"""Unit tests for corpus processing, tokenization, validation, and splitting."""

import pytest
from app.corpus import (
    LEGAL_CLAUSE_CATEGORIES,
    load_corpus,
    load_raw_corpus,
    load_sentences,
    preprocess_text,
    split_dataset,
    tokenize,
    validate_corpus_row,
)


def test_preprocess_text_normalizes_whitespace():
    raw = "  The    Receiving   Party \t shall  hold   \n "
    cleaned = preprocess_text(raw)
    assert cleaned == "The Receiving Party shall hold"


def test_preprocess_text_handles_quotes_and_dashes():
    raw = "“Confidential Information” — strictly protected ‘as-is’."
    cleaned = preprocess_text(raw)
    assert '"' in cleaned
    assert "“" not in cleaned
    assert "”" not in cleaned
    assert "-" in cleaned


def test_preprocess_text_handles_non_strings():
    assert preprocess_text(None) == ""
    assert preprocess_text(123) == ""


def test_tokenize_extracts_words():
    text = "The Buyer shall pay the Purchase Price."
    tokens = tokenize(text, lower=True)
    assert tokens == ["the", "buyer", "shall", "pay", "the", "purchase", "price"]


def test_tokenize_preserves_hyphenated_terms():
    tokens = tokenize("third-party liability pre-approved expenses", lower=True)
    assert "third-party" in tokens
    assert "pre-approved" in tokens


def test_validate_corpus_row_valid():
    valid, msg = validate_corpus_row({
        "clause_type": "Payment",
        "sentence": "Payment is due within thirty days of invoice.",
    })
    assert valid is True
    assert msg == ""


def test_validate_corpus_row_invalid_category():
    valid, msg = validate_corpus_row({
        "clause_type": "InvalidCategory",
        "sentence": "Payment is due within thirty days of invoice.",
    })
    assert valid is False
    assert "Unknown clause category" in msg


def test_validate_corpus_row_missing_sentence():
    valid, msg = validate_corpus_row({
        "clause_type": "Payment",
        "sentence": "",
    })
    assert valid is False
    assert "Missing" in msg or "short" in msg


def test_split_dataset_stratification():
    raw = load_raw_corpus()
    train_data, test_data = split_dataset(raw, test_ratio=0.2, seed=42)

    assert len(train_data) + len(test_data) == len(raw)
    # Check each category is present in test data
    test_cats = {item["clause_type"] for item in test_data}
    assert test_cats == set(LEGAL_CLAUSE_CATEGORIES)


def test_load_sentences_returns_non_empty_strings():
    sentences = load_sentences()
    assert len(sentences) > 0
    assert all(isinstance(s, str) and len(s) > 0 for s in sentences)
