"""Common utilities, configurations, and I/O handlers."""

from src.common.config import (
    DATA_DIR,
    GOLD_SUBTASK1_FILENAME,
    GOLD_SUBTASK2_FILENAME,
    MOCK_DATA_DIR,
    PREDICTION_SUBTASK1_FILENAME_TEMPLATE,
    PREDICTION_SUBTASK2_FILENAME_TEMPLATE,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SUBMISSIONS_DIR,
    SUPPORTED_LANGUAGES,
)
from src.common.io import (
    iter_jsonl,
    read_jsonl,
    validate_subtask1_gold_record,
    validate_subtask1_prediction_record,
    validate_subtask2_record,
    write_jsonl,
)
from src.common.logger import get_logger
from src.common.packaging import create_submission_zip

__all__ = [
    "DATA_DIR",
    "GOLD_SUBTASK1_FILENAME",
    "GOLD_SUBTASK2_FILENAME",
    "MOCK_DATA_DIR",
    "PREDICTION_SUBTASK1_FILENAME_TEMPLATE",
    "PREDICTION_SUBTASK2_FILENAME_TEMPLATE",
    "PROCESSED_DATA_DIR",
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "SUBMISSIONS_DIR",
    "SUPPORTED_LANGUAGES",
    "iter_jsonl",
    "read_jsonl",
    "validate_subtask1_gold_record",
    "validate_subtask1_prediction_record",
    "validate_subtask2_record",
    "write_jsonl",
    "get_logger",
    "create_submission_zip",
]
