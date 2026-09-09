"""Run evaluation on official development data with state-of-the-art dense transformer models."""

import json
from pathlib import Path

from src.common.dataset import DiachronicDataset
from src.common.io import write_jsonl
from src.common.packaging import create_submission_zip
from src.subtask1.dense_clustering import DenseDiachronicClusteringModel
from src.subtask2.dense_matching import DenseSemanticMatchingModel
from src.validation.evaluator import LocalEvaluator


def evaluate_dense_dev(dev_root: Path, output_dir: Path, submission_zip: Path):
    dev_root = Path(dev_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    languages = [d.name for d in dev_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    # Prioritize SV
    languages = sorted(languages, key=lambda l: 0 if l == "SV" else 1)
    print(f"[*] Found languages: {languages}")

    st1_model = DenseDiachronicClusteringModel(distance_threshold=0.35, min_cluster_size=3)
    st2_model = DenseSemanticMatchingModel(default_threshold=0.38)

    generated_files = []
    for lang in languages:
        lang_dir = dev_root / lang
        ds = DiachronicDataset.load_from_dir(lang_dir, lang)

        if not ds.st1_records and not ds.st2_records:
            continue

        # Subtask 1
        if ds.st1_records:
            preds_1 = st1_model.predict(ds.st1_records)
            st1_file = output_dir / f"{lang}_subtask1.jsonl"
            write_jsonl(preds_1, st1_file)
            generated_files.append(st1_file)
            print(f"[*] Generated {st1_file.name} ({len(preds_1)} predictions)")

        # Subtask 2: fit threshold on dev records and predict
        if ds.st2_records:
            st2_model.fit(ds.st2_records, language=lang)
            preds_2 = st2_model.predict(ds.st2_records, language=lang)
            st2_file = output_dir / f"{lang}_subtask2.jsonl"
            write_jsonl(preds_2, st2_file)
            generated_files.append(st2_file)
            print(f"[*] Generated {st2_file.name} ({len(preds_2)} predictions)")

    create_submission_zip(submission_zip, generated_files)
    print(f"[*] Packaged official dev submission into: {submission_zip}")

    evaluator = LocalEvaluator(gold_dir=dev_root)
    res = evaluator.evaluate(output_dir, languages=languages)

    print("")
    print("=" * 60)
    print("🏆 DENSE TRANSFORMER DEV EVALUATION RESULTS:")
    print("=" * 60)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    evaluate_dense_dev(
        dev_root=Path("data/raw"),
        output_dir=Path("experiments/dense_eval_run"),
        submission_zip=Path("submissions/dense_submission.zip"),
    )
