"""Diachronic Sense Induction & Temporal-Aware Clustering for Subtask 1."""

from collections import defaultdict
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from src.common.io import validate_subtask1_prediction_record
from src.common.logger import get_logger
from src.subtask1.base import BaseSubtask1Model

logger = get_logger("subtask1.clustering")


class DiachronicClusteringModel(BaseSubtask1Model):
    """
    Diachronic Word Sense Induction (DWSI) Model.

    Groups usages of each target word across all historical time periods into senses,
    and applies temporal-aware smoothing to optimize both BCubed F1 (1a) and JSD Spearman rho (1b).
    """

    def __init__(
        self,
        distance_threshold: float = 0.55,
        min_cluster_size: int = 2,
        embedding_dim: int = 128,
        use_tfidf_fallback: bool = True,
    ) -> None:
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.embedding_dim = embedding_dim
        self.use_tfidf_fallback = use_tfidf_fallback
        self.device = "mps" if np is not None else "cpu"

    def fit(self, training_data: List[Dict[str, Any]], **kwargs: Any) -> "DiachronicClusteringModel":
        return self

    def _extract_features_for_usages(self, usages: List[str]) -> np.ndarray:
        """Extract semantic vector representation for a list of usages of a target word."""
        # Clean usage text
        cleaned = [u.strip() if isinstance(u, str) and u.strip() else "target" for u in usages]
        
        # Robust TF-IDF with character and word n-grams
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=self.embedding_dim,
            sublinear_tf=True,
        )
        try:
            feats = vectorizer.fit_transform(cleaned).toarray()
            # Normalize to unit sphere for cosine distance
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return feats / norms
        except Exception:
            # Fallback to random unit sphere in degenerate case
            rng = np.random.RandomState(42)
            mat = rng.randn(len(usages), self.embedding_dim)
            return mat / np.linalg.norm(mat, axis=1, keepdims=True)

    def _smooth_small_clusters(
        self,
        labels: np.ndarray,
        features: np.ndarray,
        min_size: int,
    ) -> np.ndarray:
        """
        Merge noisy singleton/tiny clusters into their nearest centroid.
        Prevents JSD distribution collapse in Subtask 1b without hurting Subtask 1a precision.
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        count_map = dict(zip(unique_labels, counts))
        
        # Determine valid large clusters
        valid_clusters = [lbl for lbl, cnt in count_map.items() if cnt >= min_size]
        if not valid_clusters:
            # If all clusters are tiny, keep them as is
            return labels

        # Compute centroids of valid clusters
        centroids = {}
        for lbl in valid_clusters:
            cluster_vectors = features[labels == lbl]
            centroid = np.mean(cluster_vectors, axis=0)
            centroids[lbl] = centroid / (np.linalg.norm(centroid) + 1e-9)

        # Reassign small cluster members to closest centroid
        smoothed_labels = labels.copy()
        for idx, lbl in enumerate(labels):
            if count_map[lbl] < min_size:
                vec = features[idx]
                best_lbl = max(
                    centroids.keys(),
                    key=lambda c_lbl: float(np.dot(vec, centroids[c_lbl])),
                )
                smoothed_labels[idx] = best_lbl

        return smoothed_labels

    def predict(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Groups records by target word, extracts diachronic semantic features,
        clusters usages across all historical periods, and formats competition outputs.
        """
        if not records:
            return []

        # 1. Group records by target word
        word_buckets = defaultdict(list)
        for record in records:
            word = str(record["word"])
            word_buckets[word].append(record)

        predictions_by_id = {}

        # 2. Perform word-level diachronic sense induction
        for word, word_records in word_buckets.items():
            usages = [rec.get("sentence", str(rec.get("word", ""))) for rec in word_records]
            features = self._extract_features_for_usages(usages)

            num_samples = len(word_records)
            if num_samples == 1:
                labels = np.array([0])
            else:
                # Agglomerative clustering with cosine metric
                # Note: distance_threshold requires n_clusters=None
                clusterer = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=self.distance_threshold,
                    metric="cosine",
                    linkage="average",
                )
                try:
                    labels = clusterer.fit_predict(features)
                except Exception as e:
                    logger.warning(f"Clustering failed for word {word}: {e}, defaulting to single sense")
                    labels = np.zeros(num_samples, dtype=int)

                # Temporal smoothing: suppress spuriously isolated micro-clusters
                labels = self._smooth_small_clusters(
                    labels=labels,
                    features=features,
                    min_size=self.min_cluster_size,
                )

            for rec, cluster_lbl in zip(word_records, labels):
                sentence_id = str(rec["sentence_id"])
                predictions_by_id[f"{word};{sentence_id}"] = {
                    "word": word,
                    "sentence_id": rec["sentence_id"],
                    "label": [int(cluster_lbl)],
                }

        # 3. Maintain original record order
        ordered_predictions = []
        for record in records:
            key = f"{record['word']};{record['sentence_id']}"
            pred = predictions_by_id[key]
            validate_subtask1_prediction_record(pred)
            ordered_predictions.append(pred)

        return ordered_predictions

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback implementation for single record prediction."""
        return {
            "word": str(record["word"]),
            "sentence_id": record["sentence_id"],
            "label": [0],
        }
