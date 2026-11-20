"""Milestone 3: GPT-2 Legal Clause Classifier.

Fine-tunes GPT-2 sequence classification on labelled legal clauses and provides
inference for predicting clause categories with confidence scores.
"""

from collections.abc import Sequence
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from app.corpus import LEGAL_CLAUSE_CATEGORIES, load_corpus

# Default model directory
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "gpt2_legal_classifier"

LABEL2ID = {cat: i for i, cat in enumerate(LEGAL_CLAUSE_CATEGORIES)}
ID2LABEL = {i: cat for i, cat in enumerate(LEGAL_CLAUSE_CATEGORIES)}


class FineTunedGPT2Classifier:
    """GPT-2 based classifier for legal contract clauses."""

    def __init__(self, model_dir: Path | str | None = None, base_model_name: str = "gpt2"):
        """Initialize classifier with saved model if present, or prepare base model."""
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self.base_model_name = base_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model: PreTrainedModel | None = None
        self.is_fine_tuned: bool = False

        # Attempt to load saved model
        if self._saved_model_exists():
            self._load_saved_model()

    def _saved_model_exists(self) -> bool:
        """Check if fine-tuned weights and config exist in model_dir."""
        if not self.model_dir.exists():
            return False
        config_file = self.model_dir / "config.json"
        has_weights = (
            (self.model_dir / "model.safetensors").exists()
            or (self.model_dir / "pytorch_model.bin").exists()
        )
        return config_file.exists() and has_weights

    def _load_saved_model(self) -> None:
        """Load fine-tuned model and tokenizer from local directory."""
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir),
            num_labels=len(LEGAL_CLAUSE_CATEGORIES),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self._configure_tokenizer()
        self.model.to(self.device)
        self.model.eval()
        self.is_fine_tuned = True

    def _configure_tokenizer(self) -> None:
        """Ensure tokenizer and model have proper pad tokens configured."""
        if self.tokenizer is not None and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model is not None and self.tokenizer is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def load_base_model(self) -> None:
        """Download or load base GPT-2 model with classification head."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_name,
                num_labels=len(LEGAL_CLAUSE_CATEGORIES),
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
        self._configure_tokenizer()
        self.model.to(self.device)

    def train(
        self,
        train_data: list[dict[str, str]] | None = None,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Fine-tune GPT-2 on the legal clause dataset using PyTorch.
        
        Args:
            train_data: Optional list of dicts with 'sentence' and 'clause_type'.
            epochs: Number of training epochs.
            batch_size: Mini-batch size (practical for CPU).
            learning_rate: AdamW learning rate.
            verbose: Whether to print training loss progress.
            
        Returns:
            Dict containing training history and loss statistics.
        """
        if train_data is None:
            corpus = load_corpus()
            train_data = corpus["train"]

        if not train_data:
            raise ValueError("Training data is empty.")

        # Ensure base model and tokenizer are initialized
        if self.model is None or self.tokenizer is None or not self.is_fine_tuned:
            self.load_base_model()

        assert self.model is not None
        assert self.tokenizer is not None

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Prepare samples
        sentences = [item["sentence"] for item in train_data]
        labels = [LABEL2ID[item["clause_type"]] for item in train_data]

        history: list[dict[str, float]] = []
        n_samples = len(sentences)

        if verbose:
            print(f"Beginning fine-tuning GPT-2 on {n_samples} legal clauses across {len(LEGAL_CLAUSE_CATEGORIES)} categories...")
            print(f"Device: {self.device} | Epochs: {epochs} | Batch size: {batch_size} | LR: {learning_rate}")

        for epoch in range(1, epochs + 1):
            # Shuffle indices each epoch
            indices = torch.randperm(n_samples).tolist()
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i : i + batch_size]
                batch_texts = [sentences[idx] for idx in batch_indices]
                batch_labels = torch.tensor(
                    [labels[idx] for idx in batch_indices],
                    dtype=torch.long,
                    device=self.device,
                )

                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                ).to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    labels=batch_labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            history.append({"epoch": epoch, "loss": avg_loss})
            if verbose:
                print(f"  Epoch {epoch}/{epochs} - Average Loss: {avg_loss:.4f}")

        self.model.eval()
        self.is_fine_tuned = True
        return {
            "status": "success",
            "epochs": epochs,
            "final_loss": history[-1]["loss"] if history else 0.0,
            "history": history,
            "trained_samples": n_samples,
        }

    def save(self, save_dir: Path | str | None = None) -> Path:
        """Save the fine-tuned model and tokenizer to disk."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Cannot save: model or tokenizer has not been initialized.")

        target_dir = Path(save_dir or self.model_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(target_dir))
        self.tokenizer.save_pretrained(str(target_dir))

        meta_path = target_dir / "classifier_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "base_model": self.base_model_name,
                    "categories": LEGAL_CLAUSE_CATEGORIES,
                    "label2id": LABEL2ID,
                    "id2label": ID2LABEL,
                    "is_fine_tuned": self.is_fine_tuned,
                },
                f,
                indent=2,
            )

        self.model_dir = target_dir
        return target_dir

    def predict(self, sentence: str) -> dict[str, Any]:
        """Classify a single legal sentence into a clause category.
        
        Args:
            sentence: Legal sentence string to classify.
            
        Returns:
            Dict containing predicted category, confidence score, and all class probabilities.
        """
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError("Input sentence must be a non-empty string.")

        if self.model is None or self.tokenizer is None:
            if self._saved_model_exists():
                self._load_saved_model()
            else:
                self.load_base_model()

        assert self.model is not None
        assert self.tokenizer is not None

        self.model.eval()
        cleaned_text = sentence.strip()

        encoded = self.tokenizer(
            cleaned_text,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze(0)

        pred_idx = int(torch.argmax(probs).item())
        predicted_category = ID2LABEL[pred_idx]
        confidence = float(probs[pred_idx].item())

        prob_dict = {
            ID2LABEL[i]: float(probs[i].item())
            for i in range(len(LEGAL_CLAUSE_CATEGORIES))
        }

        return {
            "sentence": cleaned_text,
            "predicted_category": predicted_category,
            "confidence": confidence,
            "probabilities": prob_dict,
        }

    def predict_batch(self, sentences: Sequence[str]) -> list[dict[str, Any]]:
        """Classify a batch of legal sentences."""
        return [self.predict(s) for s in sentences]
