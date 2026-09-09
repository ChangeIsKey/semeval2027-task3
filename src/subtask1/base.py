"""Abstract base class and typing for Subtask 1 (Diachronic Sense Assignment & Dynamics)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.common.io import read_jsonl, validate_subtask1_prediction_record, write_jsonl
from src.common.logger import get_logger

logger = get_logger("subtask1.base")


class BaseSubtask1Model(ABC):
    """Abstract interface for Subtask 1 models."""

    @abstractmethod
    def fit(self, training_data: List[Dict[str, Any]], **kwargs: Any) -> "BaseSubtask1Model":
        """Fit or fine-tune model on training observations."""
        raise NotImplementedError

    @abstractmethod
    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Predict sense label list for a single record.

        Expected output format:
            {"word": str, "sentence_id": str, "label": list[int | str]}
        """
        raise NotImplementedError

    def predict(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict sense labels for a list of records."""
        predictions = []
        for record in records:
            pred = self.predict_record(record)
            validate_subtask1_prediction_record(pred)
            predictions.append(pred)
        return predictions

    def predict_file(self, input_path: Path | str, output_path: Path | str) -> Path:
        """Run batch inference from an input JSONL file to an output JSONL file."""
        records = read_jsonl(input_path)
        logger.info(f"Running Subtask 1 prediction for {len(records)} records from {input_path}")
        predictions = self.predict(records)
        write_jsonl(predictions, output_path)
        logger.info(f"Saved Subtask 1 predictions to {output_path}")
        return Path(output_path)
