"""Dataset loader and joiner for official SemEval-2027 Task 3 formats."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from src.common.io import read_jsonl
from src.common.logger import get_logger

logger = get_logger("dataset_loader")


class DiachronicDataset:
    """Represents a unified dataset for a single language containing usages, definitions, and gold labels."""

    def __init__(
        self,
        language: str,
        usages: Dict[str, Dict[str, Any]],
        definitions: Dict[str, str],
        st1_records: Optional[List[Dict[str, Any]]] = None,
        st2_records: Optional[List[Dict[str, Any]]] = None,
    ):
        self.language = language
        self.usages = usages  # sentence_id -> usage dict
        self.definitions = definitions  # word -> definition string
        self.st1_records = st1_records or []
        self.st2_records = st2_records or []

    @classmethod
    def load_from_dir(cls, lang_dir: Path, language: str) -> "DiachronicDataset":
        """Load usages, definitions, and available gold task files for language directory."""
        lang_dir = Path(lang_dir)
        usages_file = lang_dir / "usages.jsonl"
        defs_file = lang_dir / "definitions.jsonl"
        st1_file = lang_dir / "subtask1.jsonl"
        st2_file = lang_dir / "subtask2.jsonl"

        usages = {}
        if usages_file.exists():
            for rec in read_jsonl(usages_file):
                sid = str(rec.get("sentence_id"))
                usages[sid] = rec

        definitions = {}
        if defs_file.exists():
            for rec in read_jsonl(defs_file):
                w = str(rec.get("word"))
                definitions[w] = rec.get("definition", "")

        st1_records = []
        if st1_file.exists():
            for rec in read_jsonl(st1_file):
                sid = str(rec.get("sentence_id"))
                # Merge usage context into record for model consumption
                enriched = dict(rec)
                if sid in usages:
                    enriched["text"] = usages[sid].get("text", "")
                    enriched["sentence"] = usages[sid].get("text", "")
                    enriched["pos"] = usages[sid].get("pos", "")
                    enriched["start"] = usages[sid].get("start", None)
                    enriched["end"] = usages[sid].get("end", None)
                    enriched["year"] = usages[sid].get("year", None)
                st1_records.append(enriched)

        st2_records = []
        if st2_file.exists():
            for rec in read_jsonl(st2_file):
                sid = str(rec.get("sentence_id"))
                w = str(rec.get("word"))
                enriched = dict(rec)
                if sid in usages:
                    enriched["text"] = usages[sid].get("text", "")
                    enriched["sentence"] = usages[sid].get("text", "")
                    enriched["year"] = usages[sid].get("year", None)
                if w in definitions:
                    enriched["target_sense_definition"] = definitions[w]
                    enriched["definition"] = definitions[w]
                st2_records.append(enriched)

        logger.info(
            f"Loaded {language}: {len(usages)} usages, {len(definitions)} defs, "
            f"{len(st1_records)} ST1 records, {len(st2_records)} ST2 records"
        )
        return cls(language, usages, definitions, st1_records, st2_records)
