"""Dual-model ensemble pipeline combining Multilingual S-BERT and native Swedish KB-BERT."""

import json
from pathlib import Path

from src.common.dataset import DiachronicDataset
from src.common.ensemble_extractor import DualModelEnsembleExtractor
from src.common.io import write_jsonl
from src.common.packaging import create_submission_zip
from src.subtask1.target_clustering import TargetTokenClusteringModel
from src.subtask2.target_matching import TargetTokenMatchingModel
from src.validation.evaluator import LocalEvaluator


def run_dual_ensemble(dev_root: Path, output_dir: Path, submission_zip: Path):
    dev_root = Path(dev_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Building Dual-Model Ensemble Extractor (Multilingual MiniLM + Native Swedish KB-BERT)...")
    ensemble_extractor = DualModelEnsembleExtractor(
        multilingual_weight=0.55,
    )

    # Load SV
    sv_dir = dev_root / "SV"
    ds = DiachronicDataset.load_from_dir(sv_dir, "SV")

    # Evaluate Subtask 1 with ensemble
    print("")
    print("=" * 60)
    print("🚀 RUNNING DUAL-MODEL ENSEMBLE CLUSTERING (SUBTASK 1)")
    print("=" * 60)
    st1_model = TargetTokenClusteringModel(
        extractor=ensemble_extractor,
        distance_threshold=0.36,
        min_cluster_size=2,
        linkage="average",
    )
    preds_st1 = st1_model.predict(ds.st1_records)
    out_st1 = output_dir / "SV_subtask1.jsonl"
    write_jsonl(preds_st1, out_st1)

    # Evaluate Subtask 2 with ensemble
    print("")
    print("=" * 60)
    print("🚀 RUNNING DUAL-MODEL ENSEMBLE MATCHING (SUBTASK 2)")
    print("=" * 60)
    st2_model = TargetTokenMatchingModel(
        extractor=ensemble_extractor,
    )
    st2_model.fit(ds.st2_records, language="SV")
    preds_st2 = st2_model.predict(ds.st2_records, language="SV")
    out_st2 = output_dir / "SV_subtask2.jsonl"
    write_jsonl(preds_st2, out_st2)

    # Complete test zip with other languages
    submission_files = [out_st1, out_st2]
    for other_lang in ["EN", "IT"]:
        o_dir = dev_root / other_lang
        if o_dir.exists():
            o_ds = DiachronicDataset.load_from_dir(o_dir, other_lang)
            if o_ds.st1_records:
                p1 = st1_model.predict(o_ds.st1_records)
                f1 = output_dir / f"{other_lang}_subtask1.jsonl"
                write_jsonl(p1, f1)
                submission_files.append(f1)
            if o_ds.st2_records:
                p2 = st2_model.predict(o_ds.st2_records, language=other_lang)
                f2 = output_dir / f"{other_lang}_subtask2.jsonl"
                write_jsonl(p2, f2)
                submission_files.append(f2)

    create_submission_zip(submission_zip, submission_files)
    print(f"[*] Packaged dual-model ensemble submission: {submission_zip}")

    evaluator = LocalEvaluator(gold_dir=dev_root)
    final_res = evaluator.evaluate(output_dir, languages=["SV"])
    print("")
    print("=" * 60)
    print("🏆 DUAL-MODEL ENSEMBLE BENCHMARK RESULTS (SV DEV):")
    print("=" * 60)
    print(json.dumps(final_res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_dual_ensemble(
        dev_root=Path("data/raw"),
        output_dir=Path("experiments/dual_ensemble_run"),
        submission_zip=Path("submissions/dual_ensemble_submission.zip"),
    )
