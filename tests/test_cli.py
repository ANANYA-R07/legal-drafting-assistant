"""Unit tests for command-line interface commands."""

from app.cli import main
from app.corpus import LEGAL_CLAUSE_CATEGORIES


def test_cli_hello(capsys):
    main(["hello"])
    out = capsys.readouterr().out
    assert "Legal Drafting Assistant is ready" in out


def test_cli_categories(capsys):
    main(["categories"])
    out = capsys.readouterr().out
    for cat in LEGAL_CLAUSE_CATEGORIES:
        assert cat in out


def test_cli_generate(capsys):
    main(["generate", "--seed", "The Seller", "--n", "2", "--max-words", "10"])
    out = capsys.readouterr().out
    assert "Generated (2-gram):" in out
    assert "the seller" in out.lower()


def test_cli_classify(capsys):
    sentence = "The Receiving Party shall not disclose Confidential Information to any third party."
    main(["classify", sentence])
    out = capsys.readouterr().out
    assert "Sentence :" in out
    assert "Category :" in out
    assert "Confidentiality" in out


def test_cli_pipeline(capsys):
    main(["pipeline", "--seed", "The Buyer"])
    out = capsys.readouterr().out
    assert "Generated Sentence :" in out
    assert "Clause Category    :" in out


def test_cli_evaluate(capsys):
    main(["evaluate"])
    out = capsys.readouterr().out
    assert "precision" in out
    assert "recall" in out
    assert "f1-score" in out


def test_cli_document(capsys):
    main(["document", "--per-clause", "1"])
    out = capsys.readouterr().out
    assert "COMMERCIAL AGREEMENT (DRAFT)" in out
    assert "SECTION 1: CONFIDENTIALITY" in out
    assert "SECTION 5: INDEMNIFICATION" in out


def test_cli_history(capsys):
    main(["history"])
    out = capsys.readouterr().out
    assert "History" in out or "No recent" in out

