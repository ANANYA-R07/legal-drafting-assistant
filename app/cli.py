"""Command-line interface for the Legal Drafting Assistant.

Supports interactive menu mode with all required options, as well as direct CLI subcommands.
Integrates N-gram text generation, GPT-2 clause classification, and persistent JSON storage.
"""

import argparse
from pathlib import Path
import sys
from typing import Any

from app.corpus import LEGAL_CLAUSE_CATEGORIES, get_categories, load_corpus, load_sentences
from app.gpt2_classifier import FineTunedGPT2Classifier
from app.ngram import NgramModel
from app.storage import Store


def get_store() -> Store:
    """Return configured key-value store for application session persistence."""
    return Store()


def print_banner() -> None:
    """Display welcome banner."""
    print("=" * 68)
    print("       LEGAL DRAFTING ASSISTANT - BANKING & FINANCE")
    print("  N-gram Next-Word Predictor & GPT-2 Clause Classification")
    print("=" * 68)


def display_menu() -> None:
    """Print the interactive main menu."""
    print("\n---------------------- MAIN MENU ----------------------")
    print("  1. Generate a legal sentence using N-gram")
    print("  2. Enter a legal sentence manually")
    print("  3. Classify a legal sentence using GPT-2")
    print("  4. Generate and automatically classify a sentence")
    print("  5. View available clause categories")
    print("  6. Exit the application")
    print("-------------------------------------------------------")


def handle_generate_sentence(ngram_model: NgramModel | None = None) -> tuple[str, NgramModel]:
    """Interactive handler for Option 1: Generate legal sentence with N-gram."""
    print("\n--- Option 1: Generate a Legal Sentence (N-gram) ---")
    if ngram_model is None:
        print("Training N-gram language model on legal corpus...")
        sentences = load_sentences()
        ngram_model = NgramModel(n=2)
        ngram_model.train(sentences)

    mode = input("Draft [1] Single Sentence (default) or [2] Complete Multi-Clause Agreement? : ").strip()

    if mode == "2":
        # Multi-clause document generation
        doc_dict = ngram_model.generate_document(sentences_per_category=1)
        formatted_doc = ngram_model.format_legal_document(doc_dict)
        print("\n" + formatted_doc)

        out_file = Path("data/generated_legal_agreement.txt")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(formatted_doc, encoding="utf-8")
        print(f"\n[Saved agreement file]: {out_file.resolve()}")

        store = get_store()
        store.put("last_document", formatted_doc)

        # First sentence of document becomes the current sentence for classification
        first_clause = doc_dict.get("Confidentiality", ["The parties agree."])[0]
        store.put("current_sentence", first_clause)
        return first_clause, ngram_model

    seed = input("Enter seed phrase (press Enter for random starter): ").strip()

    order_input = input("Select N-gram order [1=Unigram, 2=Bigram (default), 3=Trigram]: ").strip()
    order = 2
    if order_input in ("1", "2", "3"):
        order = int(order_input)
    if order != ngram_model.n:
        sentences = load_sentences()
        ngram_model = NgramModel(n=order)
        ngram_model.train(sentences)

    generated = ngram_model.generate_sentence(seed=seed, max_words=25)
    print(f"\n[Generated Sentence ({ngram_model.n}-gram)]:")
    print(f"  \"{generated}\"")

    # Display top next-word predictions
    if generated:
        words = generated.rstrip(".").split()
        context = " ".join(words[-2:]) if len(words) >= 2 else (words[-1] if words else "")
        candidates = ngram_model.predict_next(context, top_k=5)
        print(f"\n[Next-word predictions following '{context}']:")
        for word, prob in candidates:
            print(f"   • {word:<15} ({prob * 100:.2f}%)")

    # Persist in storage
    store = get_store()
    store.put("current_sentence", generated)
    history = store.get("history", [])
    history.append({"type": "generated", "text": generated, "order": ngram_model.n})
    store.put("history", history[-20:])

    return generated, ngram_model


def handle_manual_input() -> str:
    """Interactive handler for Option 2: Enter sentence manually."""
    print("\n--- Option 2: Enter a Legal Sentence Manually ---")
    while True:
        text = input("Enter your legal sentence: ").strip()
        if text:
            print(f"\nSentence recorded:\n  \"{text}\"")
            store = get_store()
            store.put("current_sentence", text)
            history = store.get("history", [])
            history.append({"type": "manual_input", "text": text})
            store.put("history", history[-20:])
            return text
        print("Error: Input cannot be empty. Please enter a valid legal sentence.")


def handle_classify_sentence(
    current_sentence: str,
    classifier: FineTunedGPT2Classifier | None = None,
) -> FineTunedGPT2Classifier:
    """Interactive handler for Option 3: Classify sentence with GPT-2."""
    print("\n--- Option 3: Classify a Legal Sentence (GPT-2) ---")
    sentence_to_classify = current_sentence

    if sentence_to_classify:
        use_curr = input(f"Use current sentence: \"{sentence_to_classify}\"? [Y/n]: ").strip().lower()
        if use_curr in ("n", "no"):
            sentence_to_classify = ""

    if not sentence_to_classify:
        sentence_to_classify = input("Enter the legal sentence to classify: ").strip()

    if not sentence_to_classify:
        print("No sentence provided. Returning to menu.")
        return classifier or FineTunedGPT2Classifier()

    if classifier is None:
        print("Loading GPT-2 classification model...")
        classifier = FineTunedGPT2Classifier()
        if not classifier.is_fine_tuned and not classifier._saved_model_exists():
            print("Model weights not yet saved. Fine-tuning now on legal clauses (CPU-compatible)...")
            classifier.train(epochs=2)
            classifier.save()

    print(f"\nAnalyzing: \"{sentence_to_classify}\"")
    result = classifier.predict(sentence_to_classify)

    print("\n[Classification Result]:")
    print(f"  Predicted Category : {result['predicted_category']}")
    print(f"  Confidence Score   : {result['confidence'] * 100:.2f}%")
    print("\n[Category Probabilities]:")
    for cat, prob in result["probabilities"].items():
        bar = "█" * int(prob * 25)
        print(f"   • {cat:<18} : {prob * 100:6.2f}% | {bar}")

    # Persist classification in store
    store = get_store()
    history = store.get("history", [])
    history.append({
        "type": "classification",
        "sentence": sentence_to_classify,
        "predicted_category": result["predicted_category"],
        "confidence": result["confidence"],
    })
    store.put("history", history[-20:])

    return classifier


def handle_generate_and_classify(
    ngram_model: NgramModel | None = None,
    classifier: FineTunedGPT2Classifier | None = None,
) -> tuple[str, NgramModel, FineTunedGPT2Classifier]:
    """Interactive handler for Option 4: Generate and auto-classify."""
    print("\n--- Option 4: Generate & Automatically Classify ---")
    if ngram_model is None:
        sentences = load_sentences()
        ngram_model = NgramModel(n=2)
        ngram_model.train(sentences)

    seed = input("Enter optional seed (e.g. 'The Buyer', 'Either party', press Enter for default): ").strip()
    generated = ngram_model.generate_sentence(seed=seed, max_words=22)

    print(f"\n[Generated Legal Sentence]:\n  \"{generated}\"")

    if classifier is None:
        print("\nLoading GPT-2 clause classifier...")
        classifier = FineTunedGPT2Classifier()
        if not classifier.is_fine_tuned and not classifier._saved_model_exists():
            print("Training classifier on dataset...")
            classifier.train(epochs=2)
            classifier.save()

    result = classifier.predict(generated)
    print("\n[Automatic Classification]:")
    print(f"  Predicted Category : {result['predicted_category']}")
    print(f"  Confidence Score   : {result['confidence'] * 100:.2f}%")
    print("\n[Category Probabilities]:")
    for cat, prob in result["probabilities"].items():
        bar = "█" * int(prob * 20)
        print(f"   • {cat:<18} : {prob * 100:6.2f}% | {bar}")

    store = get_store()
    store.put("current_sentence", generated)
    history = store.get("history", [])
    history.append({
        "type": "pipeline",
        "generated": generated,
        "predicted_category": result["predicted_category"],
        "confidence": result["confidence"],
    })
    store.put("history", history[-20:])

    return generated, ngram_model, classifier


def handle_view_categories() -> None:
    """Interactive handler for Option 5: View categories and descriptions."""
    print("\n--- Option 5: Available Legal Clause Categories ---")
    descriptions = {
        "Confidentiality": "Clauses protecting non-public, proprietary information and trade secrets.",
        "Payment": "Terms governing purchase prices, invoices, payment schedules, and currency.",
        "Termination": "Conditions and notice periods for ending or renewing the agreement.",
        "Liability": "Caps on damages, warranty disclaimers, and allocations of commercial risk.",
        "Indemnification": "Obligations to defend, hold harmless, and reimburse against third-party claims.",
    }
    corpus = load_corpus()
    for cat in LEGAL_CLAUSE_CATEGORIES:
        sample_sentence = corpus["clauses"].get(cat, [""])[0] if corpus.get("clauses") else ""
        print(f"\n  [{cat.upper()}]")
        print(f"    Description : {descriptions.get(cat, 'Legal contract provision.')}")
        if sample_sentence:
            print(f"    Sample      : \"{sample_sentence}\"")


def interactive_menu() -> None:
    """Run the main interactive terminal menu loop."""
    print_banner()
    store = get_store()
    current_sentence: str = store.get("current_sentence", "")
    ngram_model: NgramModel | None = None
    classifier: FineTunedGPT2Classifier | None = None

    while True:
        display_menu()
        choice = input("Enter your choice (1-6): ").strip()

        if choice == "1":
            current_sentence, ngram_model = handle_generate_sentence(ngram_model)
        elif choice == "2":
            current_sentence = handle_manual_input()
        elif choice == "3":
            classifier = handle_classify_sentence(current_sentence, classifier)
        elif choice == "4":
            current_sentence, ngram_model, classifier = handle_generate_and_classify(ngram_model, classifier)
        elif choice == "5":
            handle_view_categories()
        elif choice == "6":
            print("\nThank you for using Legal Drafting Assistant. Goodbye!\n")
            break
        else:
            print("Invalid choice. Please select an option between 1 and 6.")


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point supporting both subcommands and interactive menu."""
    parser = argparse.ArgumentParser(
        prog="legal-drafting-assistant",
        description="Paralegals get an assistant to draft simple legal sentences using n-gram prediction and instantly classify them into clause types with a fine-tuned GPT-2 model.",
    )
    sub = parser.add_subparsers(dest="command")

    # Command: hello (preserves original smoke test)
    sub.add_parser("hello", help="Check that everything is wired up")

    # Command: menu (runs interactive mode)
    sub.add_parser("menu", help="Launch the interactive terminal menu")

    # Command: generate
    gen_parser = sub.add_parser("generate", help="Generate a legal sentence using N-gram model")
    gen_parser.add_argument("--seed", type=str, default="", help="Starting seed words")
    gen_parser.add_argument("--n", type=int, default=2, choices=[1, 2, 3], help="N-gram order (1, 2, or 3)")
    gen_parser.add_argument("--max-words", type=int, default=25, help="Maximum words to generate")

    # Command: document (full multi-clause contract drafting)
    doc_parser = sub.add_parser("document", help="Generate a full multi-clause commercial agreement draft")
    doc_parser.add_argument("--per-clause", type=int, default=1, help="Sentences per clause category")
    doc_parser.add_argument("--output", type=str, default="data/generated_legal_agreement.txt", help="Path to save generated agreement file")

    # Command: classify
    cls_parser = sub.add_parser("classify", help="Classify a legal sentence into a clause type")
    cls_parser.add_argument("sentence", type=str, nargs="?", default="", help="Sentence text to classify")

    # Command: evaluate
    sub.add_parser("evaluate", help="Run evaluation on held-out legal test clauses")

    # Command: pipeline
    pipe_parser = sub.add_parser("pipeline", help="Generate and classify a sentence automatically")
    pipe_parser.add_argument("--seed", type=str, default="The Buyer", help="Starting seed text")

    # Command: categories
    sub.add_parser("categories", help="List supported legal clause categories")

    # Command: history
    sub.add_parser("history", help="View recent drafting and classification history")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Legal Drafting Assistant is ready. Open README.md for your milestones.")
    elif args.command == "menu":
        interactive_menu()
    elif args.command == "generate":
        sentences = load_sentences()
        model = NgramModel(n=args.n)
        model.train(sentences)
        text = model.generate_sentence(seed=args.seed, max_words=args.max_words)
        print(f"Generated ({args.n}-gram): {text}")
        store = get_store()
        store.put("current_sentence", text)
    elif args.command == "document":
        sentences = load_sentences()
        model = NgramModel(n=2)
        model.train(sentences)
        doc = model.generate_document(sentences_per_category=args.per_clause)
        formatted = model.format_legal_document(doc)
        print(formatted)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(formatted, encoding="utf-8")
        print(f"\n[Draft Agreement Saved To]: {out_path.resolve()}")
        store = get_store()
        store.put("last_document", formatted)
    elif args.command == "classify":
        if not args.sentence:
            print("Error: Please provide a sentence to classify.")
            return
        classifier = FineTunedGPT2Classifier()
        if not classifier.is_fine_tuned and not classifier._saved_model_exists():
            classifier.train(epochs=1, verbose=False)
            classifier.save()
        res = classifier.predict(args.sentence)
        print(f"Sentence : \"{res['sentence']}\"")
        print(f"Category : {res['predicted_category']} ({res['confidence'] * 100:.2f}%)")
    elif args.command == "evaluate":
        from app.evaluation import evaluate_model
        res = evaluate_model()
        print(res["classification_report_text"])
    elif args.command == "pipeline":
        sentences = load_sentences()
        model = NgramModel(n=2)
        model.train(sentences)
        text = model.generate_sentence(seed=args.seed, max_words=20)
        classifier = FineTunedGPT2Classifier()
        if not classifier.is_fine_tuned and not classifier._saved_model_exists():
            classifier.train(epochs=1, verbose=False)
            classifier.save()
        res = classifier.predict(text)
        print(f"Generated Sentence : \"{text}\"")
        print(f"Clause Category    : {res['predicted_category']} (Confidence: {res['confidence'] * 100:.2f}%)")
    elif args.command == "categories":
        for cat in LEGAL_CLAUSE_CATEGORIES:
            print(f"- {cat}")
    elif args.command == "history":
        store = get_store()
        history = store.get("history", [])
        if not history:
            print("No recent drafting history found in store.")
        else:
            print(f"\nRecent History ({len(history)} entries):")
            for i, h in enumerate(history, 1):
                print(f"  {i}. [{h.get('type')}] {h.get('text') or h.get('sentence') or h.get('generated')}")
    else:
        # Default behavior with no arguments is to launch interactive menu
        interactive_menu()


if __name__ == "__main__":
    main()
