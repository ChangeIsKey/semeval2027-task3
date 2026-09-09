"""Data importer & integrity auditor for official SemEval-2027 Task 3 datasets."""

import argparse
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from src.common.config import ALLOWED_LANGUAGES, DATA_DIR, RAW_DATA_DIR
from src.common.io import read_jsonl
from src.common.logger import get_logger

logger = get_logger("data_importer")


def audit_language_data(lang_dir: Path, lang: str):
    """Audits subtask1.jsonl and subtask2.jsonl for a given language."""
    st1_file = lang_dir / "subtask1.jsonl"
    st2_file = lang_dir / "subtask2.jsonl"

    stats = {
        "language": lang,
        "st1_records": 0,
        "st1_words": 0,
        "st1_periods": 0,
        "st2_records": 0,
        "st2_positive_ratio": 0.0,
    }

    if st1_file.exists():
        records = list(read_jsonl(st1_file))
        stats["st1_records"] = len(records)
        words = {r.get("word") for r in records if "word" in r}
        periods = {r.get("period_label") for r in records if "period_label" in r}
        stats["st1_words"] = len(words)
        stats["st1_periods"] = len(periods)

    if st2_file.exists():
        records = list(read_jsonl(st2_file))
        stats["st2_records"] = len(records)
        labels = [r.get("label") for r in records if "label" in r]
        if labels:
            stats["st2_positive_ratio"] = round(sum(1 for l in labels if l == 1) / len(labels), 4)

    return stats


def import_data(source_path: Path, target_dir: Path = RAW_DATA_DIR):
    """Imports and validates official datasets from folder or zip archive."""
    source_path = Path(source_path)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    extract_tmp = None
    if source_path.is_file() and source_path.suffix == ".zip":
        logger.info(f"Extracting zip archive: {source_path}")
        extract_tmp = target_dir / "_tmp_extract"
        extract_tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_path, "r") as zf:
            zf.extractall(extract_tmp)
        scan_root = extract_tmp
    elif source_path.is_dir():
        scan_root = source_path
    else:
        raise ValueError(f"Source must be existing directory or zip: {source_path}")

    logger.info(f"Scanning for competition languages in {scan_root}")
    found_langs = []
    
    # Check directly inside scan_root or one level deep
    candidates = list(scan_root.glob("**/subtask1.jsonl"))
    for cand in candidates:
        lang_candidate = cand.parent.name.upper()
        if lang_candidate in ALLOWED_LANGUAGES:
            dest = target_dir / lang_candidate
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(cand, dest / "subtask1.jsonl")
            st2 = cand.parent / "subtask2.jsonl"
            if st2.exists():
                shutil.copy(st2, dest / "subtask2.jsonl")
            found_langs.append(lang_candidate)
            logger.info(f"Imported language: {lang_candidate}")

    if extract_tmp and extract_tmp.exists():
        shutil.rmtree(extract_tmp)

    print("")
    print("=" * 60)
    print("📋 DATA AUDIT & INSPECTION REPORT")
    print("=" * 60)
    found_langs = sorted(set(found_langs))
    for lang in found_langs:
        info = audit_language_data(target_dir / lang, lang)
        print(f"[{lang}] ST1: {info['st1_records']} samples | {info['st1_words']} words | {info['st1_periods']} periods")
        print(f"      ST2: {info['st2_records']} samples | Pos Ratio: {info['st2_positive_ratio']}")

    if not found_langs:
        print("[!] No standard language directories found. Supported: " + ", ".join(ALLOWED_LANGUAGES))
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import and audit SemEval-2027 Task 3 datasets.")
    parser.add_argument("source", type=Path, help="Path to zip file or uncompressed directory")
    parser.add_argument("--dest", type=Path, default=RAW_DATA_DIR, help="Destination directory (default: data/raw)")
    args = parser.parse_args()
    import_data(args.source, args.dest)
