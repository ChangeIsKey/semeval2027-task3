"""Validation module encapsulating official evaluation logic."""

import json
from pathlib import Path
from typing import Any, Dict

from evaluation import evaluate_subtask1, evaluate_subtask2, add_averages


class LocalEvaluator:
    """Evaluates prediction files against ground truth folders matching official SemEval2027 Task 3."""

    def __init__(self, gold_dir: Path):
        self.gold_dir = Path(gold_dir)

    def evaluate(self, predictions_dir: Path, languages: list[str] | None = None) -> Dict[str, Any]:
        """
        Run official evaluation on predictions extracted or stored in predictions_dir.
        Returns the exact evaluation JSON structure.
        """
        predictions_dir = Path(predictions_dir)
        if languages is None:
            # Detect languages from files
            subtask1_files = list(predictions_dir.glob("*_subtask1.jsonl"))
            subtask2_files = list(predictions_dir.glob("*_subtask2.jsonl"))
            languages = sorted(
                {f.name.split("_")[0] for f in subtask1_files + subtask2_files}
            )

        evaluation: Dict[str, Any] = {
            "results": {
                "subtask1a": {},
                "subtask1b": {},
                "subtask2": {},
            },
            "errors": [],
            "logs": [],
        }

        for lang in languages:
            # Subtask 1
            gold_st1 = self.gold_dir / lang / "subtask1.jsonl"
            pred_st1 = predictions_dir / f"{lang}_subtask1.jsonl"
            if gold_st1.exists() and pred_st1.exists():
                try:
                    res_1a, res_1b = evaluate_subtask1(gold_st1, pred_st1)
                    evaluation["results"]["subtask1a"][lang] = res_1a
                    evaluation["results"]["subtask1b"][lang] = res_1b
                except Exception as e:
                    evaluation["errors"].append({"language": lang, "subtask": "subtask1", "message": str(e)})

            # Subtask 2
            gold_st2 = self.gold_dir / lang / "subtask2.jsonl"
            pred_st2 = predictions_dir / f"{lang}_subtask2.jsonl"
            if gold_st2.exists() and pred_st2.exists():
                try:
                    res_2 = evaluate_subtask2(gold_st2, pred_st2)
                    evaluation["results"]["subtask2"][lang] = res_2
                except Exception as e:
                    evaluation["errors"].append({"language": lang, "subtask": "subtask2", "message": str(e)})

        add_averages(evaluation["results"])
        return evaluation
