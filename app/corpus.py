"""Milestone 1: Legal Text Corpus loading, preprocessing, validation, and dataset splitting."""

import csv
import json
import re
from pathlib import Path
from typing import Any

# Standard 5 legal clause categories
LEGAL_CLAUSE_CATEGORIES = [
    "Confidentiality",
    "Payment",
    "Termination",
    "Liability",
    "Indemnification",
]


def get_default_data_dir() -> Path:
    """Return the absolute path to the project's data directory."""
    # Check legal-drafting-assistant/data or current directory data
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def preprocess_text(text: str) -> str:
    """Clean and normalize legal text.
    
    Performs:
    - Whitespace normalization
    - Removal of unprintable/control characters
    - Normalization of quotes and dashes
    - Stripping leading and trailing whitespace
    """
    if not isinstance(text, str):
        return ""
    # Replace curly quotes and typographic dashes
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("—", "-").replace("–", "-")
    # Normalize multiple whitespace to single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str, lower: bool = True) -> list[str]:
    """Tokenize a sentence into words for n-gram modeling.
    
    Keeps words and hyphenated terms, removing extraneous punctuation.
    """
    cleaned = preprocess_text(text)
    if lower:
        cleaned = cleaned.lower()
    # Match alphanumeric words, including contractions and hyphenated legal terms
    tokens = re.findall(r"\b[\w'-]+\b", cleaned)
    return tokens


def validate_corpus_row(row: dict[str, Any]) -> tuple[bool, str]:
    """Validate that a single corpus entry contains valid clause_type and sentence text."""
    clause_type = row.get("clause_type") or row.get("category")
    sentence = row.get("sentence") or row.get("text")

    if not clause_type or not isinstance(clause_type, str):
        return False, "Missing or invalid clause_type"
    clause_type = clause_type.strip()
    if clause_type not in LEGAL_CLAUSE_CATEGORIES:
        return False, f"Unknown clause category: {clause_type}"
    if not sentence or not isinstance(sentence, str):
        return False, "Missing or invalid sentence text"
    cleaned = preprocess_text(sentence)
    if len(cleaned.split()) < 3:
        return False, "Sentence too short (under 3 words)"
    return True, ""


def load_raw_corpus(csv_path: Path | str | None = None) -> list[dict[str, str]]:
    """Load and validate raw legal corpus from a CSV file.
    
    Handles missing rows, corrupted fields, and unknown categories gracefully.
    """
    if csv_path is None:
        data_dir = get_default_data_dir()
        csv_path = data_dir / "legal_corpus.csv"
        if not csv_path.exists():
            # Check fallback in parent directory
            alt_path = Path(__file__).resolve().parent.parent.parent / "legal_corpus.csv"
            if alt_path.exists():
                csv_path = alt_path

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Corpus file not found at: {csv_path}")

    valid_rows: list[dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_valid, _ = validate_corpus_row(row)
            if is_valid:
                valid_rows.append({
                    "clause_type": (row.get("clause_type") or row.get("category", "")).strip(),
                    "sentence": preprocess_text(row.get("sentence") or row.get("text", "")),
                })
    return valid_rows


def split_dataset(
    data: list[dict[str, str]],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Perform a deterministic stratified train-test split across clause categories."""
    import random
    rng = random.Random(seed)

    by_category: dict[str, list[dict[str, str]]] = {cat: [] for cat in LEGAL_CLAUSE_CATEGORIES}
    for item in data:
        cat = item["clause_type"]
        if cat in by_category:
            by_category[cat].append(item)

    train_data: list[dict[str, str]] = []
    test_data: list[dict[str, str]] = []

    for cat, items in by_category.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_ratio))
        test_items = shuffled[:n_test]
        train_items = shuffled[n_test:]
        test_data.extend(test_items)
        train_data.extend(train_items)

    rng.shuffle(train_data)
    rng.shuffle(test_data)
    return train_data, test_data


def load_corpus(path: Path | str | None = None) -> dict[str, Any]:
    """Load, validate, process, and organize the legal corpus into structured dictionary.
    
    Returns:
        dict with keys:
            - "clauses": dict mapping category to list of sentences
            - "sentences": list of all cleaned sentences
            - "train": list of train dicts [{"clause_type": ..., "sentence": ...}]
            - "test": list of test dicts
            - "categories": list of available categories
            - "metadata": statistics about the dataset
    """
    raw_data = load_raw_corpus(path)
    train_data, test_data = split_dataset(raw_data, test_ratio=0.2, seed=42)

    clauses: dict[str, list[str]] = {cat: [] for cat in LEGAL_CLAUSE_CATEGORIES}
    all_sentences: list[str] = []

    for item in raw_data:
        clauses[item["clause_type"]].append(item["sentence"])
        all_sentences.append(item["sentence"])

    data_dir = get_default_data_dir()
    corpus_dict: dict[str, Any] = {
        "categories": LEGAL_CLAUSE_CATEGORIES,
        "clauses": clauses,
        "sentences": all_sentences,
        "train": train_data,
        "test": test_data,
        "metadata": {
            "total_sentences": len(all_sentences),
            "train_count": len(train_data),
            "test_count": len(test_data),
            "categories_count": len(LEGAL_CLAUSE_CATEGORIES),
            "per_category_counts": {cat: len(clauses[cat]) for cat in LEGAL_CLAUSE_CATEGORIES},
        },
    }

    # Save processed corpus in data folder
    processed_path = data_dir / "processed_corpus.json"
    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(corpus_dict, f, indent=2)

    return corpus_dict


def load_sentences(path: Path | str | None = None) -> list[str]:
    """Convenience helper to retrieve all preprocessed sentences for n-gram modeling."""
    corpus = load_corpus(path)
    return corpus.get("sentences", [])


def get_categories() -> list[str]:
    """Return the official list of 5 legal clause categories."""
    return list(LEGAL_CLAUSE_CATEGORIES)
