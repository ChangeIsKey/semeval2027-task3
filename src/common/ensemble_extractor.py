"""Dual-Model Ensemble Extractor combining Multilingual S-BERT and Native KB-BERT."""

from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F

from src.common.extractor import TargetTokenEmbeddingExtractor
from src.common.logger import get_logger

logger = get_logger("ensemble_extractor")


class DualModelEnsembleExtractor:
    """
    Ensemble extractor combining general multilingual semantic representations
    with native historical language representations (e.g. KB/bert-base-swedish-cased).
    """

    def __init__(
        self,
        multilingual_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        native_model: str = "KB/bert-base-swedish-cased",
        multilingual_weight: float = 0.50,
        device: Optional[str] = None,
    ) -> None:
        self.multilingual_weight = multilingual_weight
        logger.info(f"Initializing Multilingual Extractor ({multilingual_model})...")
        self.ext_multi = TargetTokenEmbeddingExtractor(
            model_name=multilingual_model,
            target_weight=0.80,
            device=device,
        )
        logger.info(f"Initializing Native Swedish KB-BERT Extractor ({native_model})...")
        self.ext_native = TargetTokenEmbeddingExtractor(
            model_name=native_model,
            target_weight=0.85,
            device=device,
        )

    def encode_records(self, records: List[Dict[str, Any]]) -> np.ndarray:
        """Extract blended ensemble embeddings for target records."""
        vecs_multi = self.ext_multi.encode_records(records)
        vecs_native = self.ext_native.encode_records(records)

        w1 = np.sqrt(self.multilingual_weight)
        w2 = np.sqrt(1.0 - self.multilingual_weight)

        concatenated = np.hstack([w1 * vecs_multi, w2 * vecs_native])
        norm = np.linalg.norm(concatenated, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return concatenated / norm

    def encode_definitions(self, definitions: List[str]) -> np.ndarray:
        """Extract blended ensemble embeddings for sense definitions."""
        vecs_multi = self.ext_multi.encode_definitions(definitions)
        vecs_native = self.ext_native.encode_definitions(definitions)

        w1 = np.sqrt(self.multilingual_weight)
        w2 = np.sqrt(1.0 - self.multilingual_weight)

        concatenated = np.hstack([w1 * vecs_multi, w2 * vecs_native])
        norm = np.linalg.norm(concatenated, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return concatenated / norm
