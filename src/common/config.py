"""Project configuration and path definitions for SemEval-2027 Task 3."""

from pathlib import Path
from typing import Final, Set

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent

# Core directories
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
MOCK_DATA_DIR: Final[Path] = DATA_DIR / "mock"
SUBMISSIONS_DIR: Final[Path] = PROJECT_ROOT / "submissions"
SCRIPTS_DIR: Final[Path] = PROJECT_ROOT / "scripts"

# Official supported languages
SUPPORTED_LANGUAGES: Final[Set[str]] = {"EN", "SV", "RU", "NL", "IT", "ES"}
ALLOWED_LANGUAGES: Final[Set[str]] = SUPPORTED_LANGUAGES

# Official file naming templates
PREDICTION_SUBTASK1_FILENAME_TEMPLATE: Final[str] = "{lang}_subtask1.jsonl"
PREDICTION_SUBTASK2_FILENAME_TEMPLATE: Final[str] = "{lang}_subtask2.jsonl"
GOLD_SUBTASK1_FILENAME: Final[str] = "subtask1.jsonl"
GOLD_SUBTASK2_FILENAME: Final[str] = "subtask2.jsonl"
