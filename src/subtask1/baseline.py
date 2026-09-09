"""Baseline models for Subtask 1."""

import hashlib
from typing import Any, Dict, List, Optional
from src.subtask1.base import BaseSubtask1Model


class BaselineSenseAssigner(BaseSubtask1Model):
    """A baseline sense assignment model.

    Partitions instances per target word into discrete sense clusters based
    on a configurable number of senses, using deterministic hashing or round-robin
    to ensure reproducibility.
    """

    def __init__(self, num_senses: int = 3, strategy: str = "hash") -> None:
        self.num_senses = max(1, num_senses)
        self.strategy = strategy
        self._fitted = False

    def fit(self, training_data: List[Dict[str, Any]], **kwargs: Any) -> "BaselineSenseAssigner":
        """Record-based fitting if training data is provided."""
        self._fitted = True
        return self

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Assigns a sense label list to the record."""
        word = str(record["word"])
        sentence_id = str(record["sentence_id"])

        if self.strategy == "hash":
            # Hash sentence_id + word to produce a consistent sense id in [0, num_senses - 1]
            digest = hashlib.md5(f"{word}:{sentence_id}".encode("utf-8")).hexdigest()
            sense_idx = int(digest, 16) % self.num_senses
            label = [sense_idx]
        elif self.strategy == "single":
            label = [0]
        else:
            label = [0]

        return {
            "word": word,
            "sentence_id": record["sentence_id"],
            "label": label,
        }
