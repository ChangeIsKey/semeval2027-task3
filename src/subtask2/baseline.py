"""Baseline models for Subtask 2."""

import hashlib
from typing import Any, Dict, List, Optional
from src.subtask2.base import BaseSubtask2Model


class BaselineChangeDetector(BaseSubtask2Model):
    """A baseline change detection model for Subtask 2.

    Outputs binary labels (0 or 1) based on a configured default mode
    or deterministic hashing for reproducibility.
    """

    def __init__(self, mode: str = "hash", threshold: float = 0.5) -> None:
        self.mode = mode
        self.threshold = threshold
        self._fitted = False

    def fit(self, training_data: List[Dict[str, Any]], **kwargs: Any) -> "BaselineChangeDetector":
        """Record-based fitting if training data is provided."""
        self._fitted = True
        return self

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Predicts binary label (0 or 1) for the target record."""
        word = str(record["word"])
        sentence_id = str(record["sentence_id"])

        if self.mode == "constant_zero":
            label = 0
        elif self.mode == "constant_one":
            label = 1
        elif self.mode == "hash":
            # Hash to deterministic float in [0, 1)
            digest = hashlib.md5(f"{word}:{sentence_id}".encode("utf-8")).hexdigest()
            val = (int(digest[:8], 16)) / 0xFFFFFFFF
            label = 1 if val >= self.threshold else 0
        else:
            label = 0

        return {
            "word": word,
            "sentence_id": record["sentence_id"],
            "label": int(label),
        }
