"""Multilingual Model Factory and Native Model Registry for SemEval-2027 Task 3."""

from typing import Dict, Optional
from src.common.ensemble_extractor import DualModelEnsembleExtractor
from src.common.extractor import TargetTokenEmbeddingExtractor
from src.common.logger import get_logger

logger = get_logger("model_factory")

# High-performance native model mapping for the 6 official competition languages
NATIVE_LANGUAGE_MODELS: Dict[str, str] = {
    "SV": "KB/bert-base-swedish-cased",
    "EN": "sentence-transformers/all-MiniLM-L6-v2",
    "IT": "dbmdz/bert-base-italian-cased",
    "ES": "PlanTL-GOB-ES/roberta-base-bne",
    "NL": "GroNLP/bert-base-dutch-cased",
    "RU": "DeepPavlov/rubert-base-cased",
}

DEFAULT_MULTILINGUAL_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class MultilingualModelFactory:
    """Factory to create optimal language-specific single or dual-ensemble extractors."""

    @staticmethod
    def get_extractor(
        language: str,
        use_ensemble: bool = True,
        device: Optional[str] = None,
    ) -> TargetTokenEmbeddingExtractor | DualModelEnsembleExtractor:
        """Returns the optimal extractor for a given language."""
        lang_upper = language.upper()
        native_model = NATIVE_LANGUAGE_MODELS.get(lang_upper)

        if use_ensemble and native_model and lang_upper == "SV":
            logger.info(f"Using Dual-Model Ensemble for language={lang_upper} ({native_model} + Multilingual)")
            return DualModelEnsembleExtractor(
                multilingual_model=DEFAULT_MULTILINGUAL_MODEL,
                native_model=native_model,
                multilingual_weight=0.55,
                device=device,
            )

        logger.info(f"Using Unified Multilingual Target Token Extractor for language={lang_upper}")
        return TargetTokenEmbeddingExtractor(
            model_name=DEFAULT_MULTILINGUAL_MODEL,
            target_weight=0.80,
            device=device,
        )
