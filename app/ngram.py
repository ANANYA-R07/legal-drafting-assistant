"""Milestone 2: N-gram Next-Word Predictor and Sentence Generator from scratch.

Supports Unigram (n=1), Bigram (n=2), and Trigram (n=3) language models
with frequency calculation, conditional probabilities, backoff smoothing,
next-word prediction, and legal sentence generation.
"""

from collections import Counter, defaultdict
import math
import random
from typing import Any

from app.corpus import tokenize

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "<unk>"


class NgramModel:
    """An N-gram Language Model built from scratch for legal text."""

    def __init__(self, n: int = 2, smoothing: float = 0.01):
        """Initialize N-gram model.
        
        Args:
            n: Order of the model (1 for Unigram, 2 for Bigram, 3 for Trigram).
            smoothing: Additive smoothing factor (Laplace / Lidstone) for unknown combinations.
        """
        if n not in (1, 2, 3):
            raise ValueError(f"Supported n-gram orders are 1, 2, or 3. Received: {n}")
        self.n = n
        self.smoothing = smoothing

        # Frequency tables
        self.unigram_counts: Counter[str] = Counter()
        self.bigram_counts: Counter[tuple[str, str]] = Counter()
        self.trigram_counts: Counter[tuple[str, str, str]] = Counter()

        # Context -> Counter[next_word] mappings for fast lookup
        self.bigram_context_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.trigram_context_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

        # Sentence starters
        self.starter_counts: Counter[str] = Counter()

        self.total_tokens: int = 0
        self.vocabulary: set[str] = set()
        self.is_trained: bool = False

    def train(self, sentences: list[str]) -> None:
        """Train the n-gram model on a list of raw or preprocessed sentences."""
        if not sentences:
            raise ValueError("Training dataset cannot be empty.")

        self.unigram_counts.clear()
        self.bigram_counts.clear()
        self.trigram_counts.clear()
        self.bigram_context_counts.clear()
        self.trigram_context_counts.clear()
        self.starter_counts.clear()
        self.vocabulary.clear()
        self.total_tokens = 0

        for sentence in sentences:
            tokens = tokenize(sentence, lower=True)
            if not tokens:
                continue

            self.starter_counts[tokens[0]] += 1

            # Pad tokens according to model order
            # For trigram: <s> <s> w1 w2 ... wn </s>
            # For bigram:  <s> w1 w2 ... wn </s>
            # For unigram: w1 w2 ... wn
            padded = [START_TOKEN] * (max(1, self.n - 1)) + tokens + [END_TOKEN]

            for token in tokens:
                self.unigram_counts[token] += 1
                self.vocabulary.add(token)
                self.total_tokens += 1

            # Bigram counts
            for i in range(len(padded) - 1):
                w1, w2 = padded[i], padded[i + 1]
                self.bigram_counts[(w1, w2)] += 1
                self.bigram_context_counts[w1][w2] += 1

            # Trigram counts
            for i in range(len(padded) - 2):
                w1, w2, w3 = padded[i], padded[i + 1], padded[i + 2]
                self.trigram_counts[(w1, w2, w3)] += 1
                self.trigram_context_counts[(w1, w2)][w3] += 1

        self.is_trained = True

    def get_unigram_prob(self, word: str) -> float:
        """Calculate unigram probability P(w) with Lidstone smoothing."""
        if self.total_tokens == 0:
            return 0.0
        vocab_size = max(1, len(self.vocabulary))
        count = self.unigram_counts.get(word, 0)
        return (count + self.smoothing) / (self.total_tokens + self.smoothing * vocab_size)

    def get_bigram_prob(self, w2: str, w1: str) -> float:
        """Calculate conditional probability P(w2 | w1)."""
        c_w1 = sum(self.bigram_context_counts[w1].values())
        if c_w1 == 0:
            # Backoff to unigram
            return self.get_unigram_prob(w2)
        c_w1_w2 = self.bigram_counts.get((w1, w2), 0)
        vocab_size = max(1, len(self.vocabulary))
        return (c_w1_w2 + self.smoothing) / (c_w1 + self.smoothing * vocab_size)

    def get_trigram_prob(self, w3: str, w1: str, w2: str) -> float:
        """Calculate conditional probability P(w3 | w1, w2)."""
        c_w1_w2 = sum(self.trigram_context_counts[(w1, w2)].values())
        if c_w1_w2 == 0:
            # Backoff to bigram
            return self.get_bigram_prob(w3, w2)
        c_w1_w2_w3 = self.trigram_counts.get((w1, w2, w3), 0)
        vocab_size = max(1, len(self.vocabulary))
        return (c_w1_w2_w3 + self.smoothing) / (c_w1_w2 + self.smoothing * vocab_size)

    def predict_next(self, prefix: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Predict the most likely next words given a prefix string.
        
        Args:
            prefix: The input seed phrase or sentence.
            top_k: Number of candidate words to return.
            
        Returns:
            List of (word, probability) sorted descending by probability.
        """
        if not self.is_trained:
            raise RuntimeError("NgramModel must be trained before predicting next words.")

        tokens = tokenize(prefix, lower=True) if prefix else []

        # Case 1: Empty input -> return most common sentence starters
        if not tokens:
            if self.starter_counts:
                total_starters = sum(self.starter_counts.values())
                candidates = [
                    (w, count / total_starters)
                    for w, count in self.starter_counts.most_common(top_k)
                ]
                return candidates
            return [(w, count / self.total_tokens) for w, count in self.unigram_counts.most_common(top_k)]

        # Candidate pool: words seen after this context, plus top unigram words
        candidate_words: set[str] = set()

        if self.n == 3:
            if len(tokens) >= 2:
                w1, w2 = tokens[-2], tokens[-1]
                seen_trigram = self.trigram_context_counts.get((w1, w2))
                if seen_trigram:
                    candidate_words.update(seen_trigram.keys())
            if not candidate_words and len(tokens) >= 1:
                # Backoff to bigram context
                w2 = tokens[-1]
                seen_bigram = self.bigram_context_counts.get(w2)
                if seen_bigram:
                    candidate_words.update(seen_bigram.keys())

        elif self.n == 2:
            w = tokens[-1]
            seen_bigram = self.bigram_context_counts.get(w)
            if seen_bigram:
                candidate_words.update(seen_bigram.keys())

        # If still no candidates found (unknown word or unseen context), back off to unigrams
        if not candidate_words:
            candidate_words.update([w for w, _ in self.unigram_counts.most_common(20)])
            candidate_words.add(END_TOKEN)

        # Filter out START_TOKEN
        candidate_words.discard(START_TOKEN)

        # Score candidates
        scored: list[tuple[str, float]] = []
        for word in candidate_words:
            if self.n == 3 and len(tokens) >= 2:
                prob = self.get_trigram_prob(word, tokens[-2], tokens[-1])
            elif self.n >= 2 and len(tokens) >= 1:
                prob = self.get_bigram_prob(word, tokens[-1])
            else:
                prob = self.get_unigram_prob(word)
            scored.append((word, prob))

        # Normalize probabilities over returned candidates
        total_p = sum(p for _, p in scored) or 1.0
        normalized = [(w, p / total_p) for w, p in scored]
        normalized.sort(key=lambda item: item[1], reverse=True)
        return normalized[:top_k]

    def generate_sentence(
        self,
        seed: str = "",
        max_words: int = 25,
        temperature: float = 0.7,
        seed_random: int | None = None,
    ) -> str:
        """Generate a complete legal-style sentence.
        
        Args:
            seed: Optional starting words.
            max_words: Maximum words to generate.
            temperature: Sampling temperature. 0.0 = greedy argmax, >0.0 = probabilistic.
            seed_random: Optional random seed for reproducible generation.
            
        Returns:
            A clean, properly capitalized and punctuated legal sentence.
        """
        if not self.is_trained:
            raise RuntimeError("NgramModel must be trained before generating text.")

        rng = random.Random(seed_random) if seed_random is not None else random

        tokens: list[str] = tokenize(seed, lower=True) if seed else []

        # If no seed provided, pick a starter
        if not tokens:
            if self.starter_counts:
                starters, counts = zip(*self.starter_counts.items())
                tokens.append(rng.choices(starters, weights=counts, k=1)[0])
            elif self.vocabulary:
                tokens.append(rng.choice(list(self.vocabulary)))
            else:
                return "The parties agree to the terms herein."

        while len(tokens) < max_words:
            candidates = self.predict_next(" ".join(tokens), top_k=10)
            if not candidates:
                break

            words, probs = zip(*candidates)

            # Check if END_TOKEN is among options
            if temperature <= 0.001:
                # Greedy choice
                next_word = words[0]
            else:
                # Apply temperature
                scaled_probs = [math.pow(max(p, 1e-9), 1.0 / max(temperature, 0.1)) for p in probs]
                total_scale = sum(scaled_probs) or 1.0
                norm_probs = [p / total_scale for p in scaled_probs]
                next_word = rng.choices(words, weights=norm_probs, k=1)[0]

            if next_word == END_TOKEN:
                break
            tokens.append(next_word)

        if not tokens:
            return "The parties agree to the terms herein."

        # Capitalize and format sentence
        text = " ".join(tokens)
        text = text[0].upper() + text[1:]
        if not text.endswith((".", ";", ":")):
            text += "."
        return text

    def generate(self, prompt: str = "", max_words: int = 25, temperature: float = 0.7) -> str:
        """Alias for generate_sentence to support both API styles."""
        return self.generate_sentence(seed=prompt, max_words=max_words, temperature=temperature)

    def generate_document(
        self,
        categories: list[str] | None = None,
        sentences_per_category: int = 1,
        temperature: float = 0.7,
    ) -> dict[str, list[str]]:
        """Generate a multi-clause legal agreement draft across specified categories.
        
        Args:
            categories: List of clause categories to include.
            sentences_per_category: Number of clauses to generate per category.
            temperature: Sampling temperature for generation.
            
        Returns:
            Dict mapping category names to lists of generated legal sentences.
        """
        if not self.is_trained:
            raise RuntimeError("NgramModel must be trained before generating documents.")

        if categories is None:
            categories = ["Confidentiality", "Payment", "Termination", "Liability", "Indemnification"]

        seeds = {
            "Confidentiality": "The Receiving Party shall hold all Confidential Information",
            "Payment": "The Buyer shall pay the Purchase Price",
            "Termination": "Either party may terminate this Agreement",
            "Liability": "Neither party shall be liable for indirect",
            "Indemnification": "The Seller shall indemnify the Buyer",
        }

        document: dict[str, list[str]] = {}
        for cat in categories:
            seed = seeds.get(cat, "The parties agree that")
            clauses: list[str] = []
            for _ in range(sentences_per_category):
                clause = self.generate_sentence(seed=seed, max_words=22, temperature=temperature)
                clauses.append(clause)
            document[cat] = clauses
        return document

    def format_legal_document(
        self,
        document_dict: dict[str, list[str]],
        title: str = "COMMERCIAL AGREEMENT (DRAFT)",
    ) -> str:
        """Format generated clauses into a formal legal agreement text."""
        lines = [
            "=" * 64,
            f"           {title}",
            "=" * 64,
            "",
            "This Agreement is entered into by and between the undersigned parties.",
            "",
        ]
        section_num = 1
        for cat, sentences in document_dict.items():
            lines.append(f"SECTION {section_num}: {cat.upper()}")
            for s in sentences:
                lines.append(f"  {s}")
            lines.append("")
            section_num += 1
        lines.append("IN WITNESS WHEREOF, the parties hereto have executed this Agreement.")
        lines.append("=" * 64)
        return "\n".join(lines)

    def get_model_stats(self) -> dict[str, Any]:
        """Return operational statistics for the trained model."""
        return {
            "order": self.n,
            "order_name": {1: "Unigram", 2: "Bigram", 3: "Trigram"}.get(self.n, f"{self.n}-gram"),
            "vocabulary_size": len(self.vocabulary),
            "total_tokens": self.total_tokens,
            "unique_bigrams": len(self.bigram_counts),
            "unique_trigrams": len(self.trigram_counts),
            "is_trained": self.is_trained,
        }
