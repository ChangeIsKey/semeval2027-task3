"""Target-Token-Aware Diachronic Clustering for Subtask 1."""

from collections import defaultdict
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src.common.extractor import TargetTokenEmbeddingExtractor
from src.common.io import validate_subtask1_prediction_record
from src.common.logger import get_logger
from src.subtask1.base import BaseSubtask1Model

logger = get_logger("subtask1.target_clustering")


class TargetTokenClusteringModel(BaseSubtask1Model):
    """
    Subtask 1 Model isolating target tokens via exact character offsets,
    debiasing historical period noise, and clustering with temporal smoothing.
    """

    def __init__(
        self,
        extractor: Optional[TargetTokenEmbeddingExtractor] = None,
        distance_threshold: float = 0.32,
        min_cluster_size: int = 3,
        linkage: str = "average",
    ) -> None:
        self.extractor = extractor or TargetTokenEmbeddingExtractor()
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.linkage = linkage

    def fit(self, training_data: List[Dict[str, Any]], **kwargs: Any) -> "TargetTokenClusteringModel":
        return self

    def _smooth_clusters(self, labels: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Merge micro-clusters into nearest cluster centroid."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        count_map = dict(zip(unique_labels, counts))
        valid_labels = [lbl for lbl, cnt in count_map.items() if cnt >= self.min_cluster_size]

        if not valid_labels or len(valid_labels) == len(unique_labels):
            return labels

        centroids = {}
        for lbl in valid_labels:
            c = np.mean(features[labels == lbl], axis=0)
            centroids[lbl] = c / (np.linalg.norm(c) + 1e-9)

        smoothed = labels.copy()
        for idx, lbl in enumerate(labels):
            if count_map[lbl] < self.min_cluster_size:
                vec = features[idx]
                best_lbl = max(centroids.keys(), key=lambda c_lbl: float(np.dot(vec, centroids[c_lbl])))
                smoothed[idx] = best_lbl

        return smoothed

    def predict(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group usages by word, extract target token representations, cluster, and output."""
        if not records:
            return []

        word_buckets = defaultdict(list)
        for record in records:
            w = str(record["word"])
            word_buckets[w].append(record)

        predictions_by_id = {}

        for word, word_records in word_buckets.items():
            # 1. Extract high-precision target-token representations
            features = self.extractor.encode_records(word_records)
            n_samples = len(word_records)

            if n_samples <= 1:
                labels = np.zeros(n_samples, dtype=int)
            else:
                clusterer = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=self.distance_threshold,
                    metric="cosine",
                    linkage=self.linkage,
                )
                try:
                    labels = clusterer.fit_predict(features)
                    labels = self._smooth_clusters(labels, features)
                except Exception as e:
                    logger.warning(f"Clustering error for {word}: {e}, using single sense")
                    labels = np.zeros(n_samples, dtype=int)

            for rec, cluster_lbl in zip(word_records, labels):
                sid = str(rec["sentence_id"])
                predictions_by_id[f"{word};{sid}"] = {
                    "word": word,
                    "sentence_id": rec["sentence_id"],
                    "label": [int(cluster_lbl)],
                }

        ordered_predictions = []
        for r in records:
            key = f"{r['word']};{r['sentence_id']}"
            pred = predictions_by_id[key]
            validate_subtask1_prediction_record(pred)
            ordered_predictions.append(pred)

        return ordered_predictions

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "word": str(record["word"]),
            "sentence_id": record["sentence_id"],
            "label": [0],
        }
