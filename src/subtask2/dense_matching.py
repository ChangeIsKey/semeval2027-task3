"""Dense Semantic Matching Model for Subtask 2."""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import f1_score
import torch
from transformers import AutoModel, AutoTokenizer

from src.common.io import validate_subtask2_record
from src.common.logger import get_logger
from src.subtask2.base import BaseSubtask2Model

logger = get_logger("subtask2.dense")


class DenseSemanticMatchingModel(BaseSubtask2Model):
    """
    Subtask 2 model matching target sense definitions and usages in dense vector space.
    Computes cosine similarity of contextual embeddings and calibrates language thresholds.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        default_threshold: float = 0.50,
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.default_threshold = default_threshold
        self.batch_size = batch_size
        self.language_thresholds: Dict[str, float] = {}

        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        logger.info(f"Initializing DenseSemanticMatchingModel on device={self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.encoder.eval()

    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
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
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            all_embeddings.append(normalized.cpu().numpy())

        return np.vstack(all_embeddings)

    def fit(self, training_data: List[Dict[str, Any]], language: Optional[str] = None, **kwargs: Any) -> "DenseSemanticMatchingModel":
        """Optimize classification threshold on labeled data by maximizing Binary F1."""
        if not training_data or "label" not in training_data[0]:
            return self

        defs = [r.get("target_sense_definition", r.get("definition", "")) for r in training_data]
        usages = [r.get("text", r.get("sentence", "")) for r in training_data]
        golds = np.array([int(r["label"]) for r in training_data])

        def_vecs = self._encode_texts(defs)
        usage_vecs = self._encode_texts(usages)
        sims = np.sum(def_vecs * usage_vecs, axis=1)

        best_th = 0.50
        best_f1 = -1.0
        for th in np.linspace(0.20, 0.80, 61):
            preds = (sims >= th).astype(int)
            f1 = float(f1_score(golds, preds, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_th = float(th)

        logger.info(f"Fitted threshold for language={language}: th={best_th:.3f} (Dev F1={best_f1:.4f})")
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

    def predict(self, records: List[Dict[str, Any]], language: Optional[str] = None) -> List[Dict[str, Any]]:
        """Batch predict binary labels with dense semantic similarity."""
        if not records:
            return []

        defs = [r.get("target_sense_definition", r.get("definition", "")) for r in records]
        usages = [r.get("text", r.get("sentence", "")) for r in records]

        def_vecs = self._encode_texts(defs)
        usage_vecs = self._encode_texts(usages)
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
