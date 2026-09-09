"""Advanced pipeline: runs DiachronicClusteringModel and HypothesisMatchingModel, packages zip and validates."""

import json
from pathlib import Path

from src.common.io import read_jsonl, write_jsonl
from src.common.packaging import create_submission_zip
from src.subtask1.clustering import DiachronicClusteringModel
from src.subtask2.matching import HypothesisMatchingModel
from src.validation.evaluator import LocalEvaluator


def run_advanced_pipeline(data_root: Path, output_dir: Path, submission_zip: Path):
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    languages = [d.name for d in data_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    print(f"[*] Detected languages: {languages}")

    st1_model = DiachronicClusteringModel(distance_threshold=0.55, min_cluster_size=2)
    st2_model = HypothesisMatchingModel(default_threshold=0.20)

    generated_files = []
    for lang in languages:
        st1_gold_file = data_root / lang / "subtask1.jsonl"
        st2_gold_file = data_root / lang / "subtask2.jsonl"

        if st1_gold_file.exists():
            records = list(read_jsonl(st1_gold_file))
            preds_st1 = st1_model.predict(records)
            st1_pred_file = output_dir / f"{lang}_subtask1.jsonl"
            write_jsonl(preds_st1, st1_pred_file)
            generated_files.append(st1_pred_file)
            print(f"[*] Generated {st1_pred_file.name} ({len(preds_st1)} instances)")

        if st2_gold_file.exists():
            records = list(read_jsonl(st2_gold_file))
            preds_st2 = st2_model.predict(records, language=lang)
            st2_pred_file = output_dir / f"{lang}_subtask2.jsonl"
            write_jsonl(preds_st2, st2_pred_file)
            generated_files.append(st2_pred_file)
            print(f"[*] Generated {st2_pred_file.name} ({len(preds_st2)} instances)")

    create_submission_zip(submission_zip, generated_files)
    print(f"[*] Successfully packaged advanced submission into {submission_zip}")

    evaluator = LocalEvaluator(gold_dir=data_root)
    results = evaluator.evaluate(output_dir, languages=languages)
    print("")
    print("=" * 50)
    print("🏆 ADVANCED MODEL LOCAL VALIDATION RESULTS:")
    print("=" * 50)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    mock_dir = Path("data/mock")
    work_dir = Path("experiments/advanced_run")
    zip_path = Path("submissions/advanced_submission.zip")
    run_advanced_pipeline(mock_dir, work_dir, zip_path)
