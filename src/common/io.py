"""Serialization and validation utilities for SemEval-2027 Task 3 formats."""

import json
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List


def read_jsonl(file_path: Path | str) -> List[Dict[str, Any]]:
    """Reads a JSONL file and returns a list of dictionaries."""
    path = Path(file_path)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {err}") from err
    return records


def iter_jsonl(file_path: Path | str) -> Generator[Dict[str, Any], None, None]:
    """Iterates lazily through a JSONL file."""
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {err}") from err


def write_jsonl(records: Iterable[Dict[str, Any]], file_path: Path | str) -> None:
    """Writes an iterable of dictionaries to a JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_subtask1_prediction_record(record: Dict[str, Any]) -> None:
    """Validates a Subtask 1 prediction record against competition format."""
    required = ("word", "sentence_id", "label")
    for field in required:
        if field not in record:
            raise ValueError(f"Subtask 1 prediction record missing field '{field}': {record}")
    if not isinstance(record["word"], str) or not record["word"]:
        raise ValueError(f"'word' must be non-empty string in {record}")
    if not isinstance(record["sentence_id"], (str, int)):
        raise ValueError(f"'sentence_id' must be string or int in {record}")
    if not isinstance(record["label"], list):
        raise ValueError(f"'label' must be a list in {record}")


def validate_subtask1_gold_record(record: Dict[str, Any]) -> None:
    """Validates a Subtask 1 gold record."""
    required = ("word", "period_label", "sentence_id", "label")
    for field in required:
        if field not in record:
            raise ValueError(f"Subtask 1 gold record missing field '{field}': {record}")
    if not isinstance(record["word"], str) or not record["word"]:
        raise ValueError(f"'word' must be non-empty string in {record}")
    if not isinstance(record["period_label"], str) or not record["period_label"]:
        raise ValueError(f"'period_label' must be non-empty string in {record}")
    if not isinstance(record["sentence_id"], (str, int)):
        raise ValueError(f"'sentence_id' must be string or int in {record}")
    if not isinstance(record["label"], list):
        raise ValueError(f"'label' must be a list in {record}")


def validate_subtask2_record(record: Dict[str, Any]) -> None:
    """Validates a Subtask 2 prediction or gold record."""
    required = ("word", "sentence_id", "label")
    for field in required:
        if field not in record:
            raise ValueError(f"Subtask 2 record missing field '{field}': {record}")
    if not isinstance(record["word"], str) or not record["word"]:
        raise ValueError(f"'word' must be non-empty string in {record}")
    if not isinstance(record["sentence_id"], (str, int)):
        raise ValueError(f"'sentence_id' must be string or int in {record}")
    if record["label"] not in (0, 1):
        raise ValueError(f"'label' must be integer 0 or 1 in {record}")
