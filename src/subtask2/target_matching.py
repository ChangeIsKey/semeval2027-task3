"""Target-Token Cross-Similarity Matching Model for Subtask 2."""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import f1_score

from src.common.extractor import TargetTokenEmbeddingExtractor
from src.common.io import validate_subtask2_record
from src.common.logger import get_logger
from src.subtask2.base import BaseSubtask2Model

logger = get_logger("subtask2.target_matching")


class TargetTokenMatchingModel(BaseSubtask2Model):
    """
    Subtask 2 model matching target sense definitions and usages by comparing
    definition embeddings against exact target-token contextualized vectors.
    """

    def __init__(
        self,
        extractor: Optional[TargetTokenEmbeddingExtractor] = None,
        default_threshold: float = 0.40,
    ) -> None:
        self.extractor = extractor or TargetTokenEmbeddingExtractor()
        self.default_threshold = default_threshold
        self.language_thresholds: Dict[str, float] = {}

    def fit(
        self,
        training_data: List[Dict[str, Any]],
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> "TargetTokenMatchingModel":
        """Calibrate optimal threshold on ground truth labels."""
        if not training_data or "label" not in training_data[0]:
            return self

        defs = [r.get("target_sense_definition", r.get("definition", "")) for r in training_data]
        golds = np.array([int(r["label"]) for r in training_data])

        usage_vecs = self.extractor.encode_records(training_data)
        def_vecs = self.extractor.encode_definitions(defs)
        sims = np.sum(def_vecs * usage_vecs, axis=1)

        best_th = self.default_threshold
        best_f1 = -1.0
        for th in np.linspace(0.15, 0.75, 61):
            preds = (sims >= th).astype(int)
            f1 = float(f1_score(golds, preds, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_th = float(th)

        logger.info(f"Fitted target-token threshold for lang={language}: th={best_th:.3f} (Dev F1={best_f1:.4f})")
        if language:
            self.language_thresholds[language] = best_th
        else:
            self.default_threshold = best_th

        return self

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "word": str(record["word"]),
            "sentence_id": record["sentence_id"],
            "label": 0,
        }

    def predict(
        self,
        records: List[Dict[str, Any]],
        language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Batch predict binary change labels."""
        if not records:
            return []

        defs = [r.get("target_sense_definition", r.get("definition", "")) for r in records]
        usage_vecs = self.extractor.encode_records(records)
        def_vecs = self.extractor.encode_definitions(defs)
        sims = np.sum(def_vecs * usage_vecs, axis=1)

        th = self.language_thresholds.get(language, self.default_threshold) if language else self.default_threshold
        predictions = []
        for rec, sim in zip(records, sims):
            pred = {
                "word": str(rec["word"]),
                "sentence_id": rec["sentence_id"],
                "label": int(sim >= th),
            }
            validate_subtask2_record(pred)
            predictions.append(pred)

        return predictions
