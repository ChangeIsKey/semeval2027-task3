"""High-precision Target Token and Contextual Embedding Extractor for SemEval-2027 Task 3."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.common.logger import get_logger

logger = get_logger("extractor")


class TargetTokenEmbeddingExtractor:
    """
    Extracts contextualized representations specifically isolating target tokens
    via exact character offsets [start, end], blended with context embeddings.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        target_weight: float = 0.85,
        max_length: int = 128,
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.target_weight = target_weight
        self.max_length = max_length
        self.batch_size = batch_size

        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        logger.info(f"TargetTokenEmbeddingExtractor initialized on device={self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.encoder.eval()

    @torch.no_grad()
    def encode_records(self, records: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extracts L2-normalized target-token contextual representations.
        Each record can contain 'text', 'sentence', 'start', 'end', 'word'.
        """
        all_embeddings = []

        for i in range(0, len(records), self.batch_size):
            batch = records[i : i + self.batch_size]
            texts = [r.get("text", r.get("sentence", str(r.get("word", "")))) for r in batch]
            cleaned_texts = [t if isinstance(t, str) and t.strip() else "target" for t in texts]

            # Tokenize with fast offset mapping
            encoded = self.tokenizer(
                cleaned_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
                return_tensors="pt",
            )

            offsets = encoded.pop("offset_mapping")  # (batch, seq_len, 2)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

            # 1. Global sentence mean pooled vector
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sentence_vecs = torch.sum(last_hidden * mask_expanded, dim=1) / torch.clamp(
                mask_expanded.sum(dim=1), min=1e-9
            )

            # 2. Extract slice specifically corresponding to [start, end]
            target_vec_list = []
            for b_idx, rec in enumerate(batch):
                char_start = rec.get("start")
                char_end = rec.get("end")

                # Parse integer offsets if provided as string
                try:
                    c_s = int(char_start) if char_start is not None else None
                    c_e = int(char_end) if char_end is not None else None
                except (ValueError, TypeError):
                    c_s, c_e = None, None

                # If no valid offsets, try fallback string search
                if c_s is None or c_e is None:
                    target_w = str(rec.get("word", "")).strip().lower()
                    haystack = cleaned_texts[b_idx].lower()
                    pos = haystack.find(target_w) if target_w else -1
                    if pos != -1:
                        c_s, c_e = pos, pos + len(target_w)

                # Match token indices
                target_token_indices = []
                if c_s is not None and c_e is not None:
                    seq_offsets = offsets[b_idx].tolist()
                    for t_idx, (t_s, t_e) in enumerate(seq_offsets):
                        if (t_s, t_e) != (0, 0) and t_s < c_e and t_e > c_s:
                            target_token_indices.append(t_idx)

                if target_token_indices:
                    # Average over matching sub-tokens
                    idx_tensor = torch.tensor(target_token_indices, device=self.device)
                    sliced = last_hidden[b_idx].index_select(0, idx_tensor)
                    t_vec = torch.mean(sliced, dim=0)
                else:
                    # Fallback to sentence vector
                    t_vec = sentence_vecs[b_idx]

                # Blend target representation with global context
                blended = self.target_weight * t_vec + (1.0 - self.target_weight) * sentence_vecs[b_idx]
                target_vec_list.append(blended)

            batch_vecs = torch.stack(target_vec_list, dim=0)
            normed = F.normalize(batch_vecs, p=2, dim=1)
            all_embeddings.append(normed.cpu().numpy())

        return np.vstack(all_embeddings)

    @torch.no_grad()
    def encode_definitions(self, definitions: List[str]) -> np.ndarray:
        """Encode textual definitions to L2-normalized representations."""
        cleaned = [d if isinstance(d, str) and d.strip() else "definition" for d in definitions]
        all_vecs = []
        for i in range(0, len(cleaned), self.batch_size):
            chunk = cleaned[i : i + self.batch_size]
            encoded = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.encoder(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            pooled = torch.sum(outputs.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            normed = F.normalize(pooled, p=2, dim=1)
            all_vecs.append(normed.cpu().numpy())

        return np.vstack(all_vecs)
