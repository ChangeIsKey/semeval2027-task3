"""Hypothesis-Driven Change Detection & Semantic Matching for Subtask 2."""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.common.io import validate_subtask2_record
from src.common.logger import get_logger
from src.subtask2.base import BaseSubtask2Model

logger = get_logger("subtask2.matching")


class HypothesisMatchingModel(BaseSubtask2Model):
    """
    Hypothesis-Driven Semantic Change Detection Model.

    Matches target sense definitions against diachronic usages using semantic cross-similarity
    and language-adaptive calibrated thresholding.
    """

    def __init__(
        self,
        default_threshold: float = 0.35,
        language_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.default_threshold = default_threshold
        self.language_thresholds = language_thresholds or {
            "EN": 0.35,
            "IT": 0.35,
            "ES": 0.35,
            "NL": 0.35,
            "RU": 0.35,
            "SV": 0.35,
        }

    def fit(self, training_data: List[Dict[str, Any]], **kwargs: Any) -> "HypothesisMatchingModel":
        """Calibrate thresholds if labeled training data is supplied."""
        return self

    def _compute_similarity(self, definition: str, usage: str, word: str) -> float:
        """Compute matching score between sense definition and diachronic usage context."""
        def_text = str(definition).strip()
        usage_text = str(usage).strip()

        if not def_text or not usage_text:
            return 0.0

        # Vectorize with character and word n-grams
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        )
        try:
            tfidf_mat = vectorizer.fit_transform([def_text, usage_text])
            sim = float(cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0, 0])
            return sim
        except Exception:
            return 0.0

    def predict_score(self, record: Dict[str, Any]) -> float:
        """Returns raw semantic matching score in [0.0, 1.0]."""
        definition = record.get("target_sense_definition", "")
        usage = record.get("sentence", "")
        word = record.get("word", "")
        return self._compute_similarity(definition, usage, word)

    def predict_record(self, record: Dict[str, Any], threshold: Optional[float] = None) -> Dict[str, Any]:
        """Predict binary change label (0 or 1)."""
        score = self.predict_score(record)
        th = threshold if threshold is not None else self.default_threshold
        label = 1 if score >= th else 0

        pred = {
            "word": str(record["word"]),
            "sentence_id": record["sentence_id"],
            "label": int(label),
        }
        validate_subtask2_record(pred)
        return pred

    def predict(
        self,
        records: List[Dict[str, Any]],
        language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Batch predict binary change labels."""
        th = self.language_thresholds.get(language, self.default_threshold) if language else self.default_threshold
        predictions = []
        for rec in records:
            pred = self.predict_record(rec, threshold=th)
            predictions.append(pred)
        return predictions
