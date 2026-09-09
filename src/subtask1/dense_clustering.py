"""Transformer-based Dense Diachronic Clustering for Subtask 1."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import torch
from transformers import AutoModel, AutoTokenizer

from src.common.io import validate_subtask1_prediction_record
from src.common.logger import get_logger
from src.subtask1.base import BaseSubtask1Model

logger = get_logger("subtask1.dense")


class DenseDiachronicClusteringModel(BaseSubtask1Model):
    """
    State-of-the-art Dense Multilingual DWSI Model.
    Extracts contextualized representations using multilingual transformers,
    clusters cross-temporal usages with cosine affinity, and enforces temporal smoothing.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        distance_threshold: float = 0.28,
        min_cluster_size: int = 2,
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.batch_size = batch_size

        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        logger.info(f"Initializing DenseDiachronicClusteringModel on device={self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.encoder.eval()

    def fit(self, training_data: List[Dict[str, Any]], **kwargs: Any) -> "DenseDiachronicClusteringModel":
        return self

    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Batch encode sentences to L2-normalized dense embeddings."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = [t if isinstance(t, str) and t.strip() else "none" for t in texts[i : i + self.batch_size]]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.encoder(**inputs)
            # Mean pooling over token embeddings with attention mask
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            # L2 normalize
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            all_embeddings.append(normalized.cpu().numpy())

        return np.vstack(all_embeddings)

    def _smooth_clusters(self, labels: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Merge tiny noisy clusters (< min_cluster_size) into nearest robust centroid."""
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
        """Group records by target word, encode with transformer, cluster and regularize."""
        if not records:
            return []

        word_buckets = defaultdict(list)
        for record in records:
            w = str(record["word"])
            word_buckets[w].append(record)

        predictions_by_id = {}

        for word, word_records in word_buckets.items():
            texts = [r.get("text", r.get("sentence", str(word))) for r in word_records]
            features = self._encode_texts(texts)
            n_samples = len(word_records)

            if n_samples <= 1:
                labels = np.zeros(n_samples, dtype=int)
            else:
                clusterer = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=self.distance_threshold,
                    metric="cosine",
                    linkage="average",
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
