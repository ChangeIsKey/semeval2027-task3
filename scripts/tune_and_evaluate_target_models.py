"""Grid tuning and full evaluation for Target-Token Isolation Models on official data."""

import json
from pathlib import Path

from src.common.dataset import DiachronicDataset
from src.common.extractor import TargetTokenEmbeddingExtractor
from src.common.io import write_jsonl
from src.common.packaging import create_submission_zip
from src.subtask1.target_clustering import TargetTokenClusteringModel
from src.subtask2.target_matching import TargetTokenMatchingModel
from src.validation.evaluator import LocalEvaluator


def run_target_tuning(dev_root: Path, output_dir: Path, submission_zip: Path):
    dev_root = Path(dev_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize shared high-precision target-token extractor
    print("[*] Initializing shared TargetTokenEmbeddingExtractor on MPS...")
    extractor = TargetTokenEmbeddingExtractor(target_weight=0.80)

    # 2. Load SV benchmark
    sv_dir = dev_root / "SV"
    ds = DiachronicDataset.load_from_dir(sv_dir, "SV")

    # 3. Explore clustering distance thresholds for Pareto optimal (1a FCubed vs 1b Spearman)
    candidate_thresholds = [0.26, 0.30, 0.34, 0.38]
    best_res_1a = None
    best_res_1b = None
    best_th = candidate_thresholds[0]
    best_preds_1 = []

    evaluator = LocalEvaluator(gold_dir=dev_root)
    tmp_eval_dir = output_dir / "tuning_tmp"
    tmp_eval_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print("=" * 60)
    print("🔍 GRID TUNING SUBTASK 1 (DISTANCE THRESHOLD):")
    print("=" * 60)

    for th in candidate_thresholds:
        model_1 = TargetTokenClusteringModel(
            extractor=extractor,
            distance_threshold=th,
            min_cluster_size=2,
            linkage="average",
        )
        preds_1 = model_1.predict(ds.st1_records)
        write_jsonl(preds_1, tmp_eval_dir / "SV_subtask1.jsonl")

        # Dummy subtask2 to allow evaluator run
        write_jsonl(
            [{"word": r["word"], "sentence_id": r["sentence_id"], "label": 0} for r in ds.st2_records],
            tmp_eval_dir / "SV_subtask2.jsonl",
        )

        eval_res = evaluator.evaluate(tmp_eval_dir, languages=["SV"])
        res_1a = eval_res["results"]["subtask1a"]["SV"]
        res_1b = eval_res["results"]["subtask1b"]["SV"]

        print(
            f"[Threshold {th:.2f}] -> 1a P: {res_1a['precision']:.4f} | "
            f"1a R: {res_1a['recall']:.4f} | 1a F-cubed: {res_1a['fcubed']:.4f} | "
            f"1b Spearman rho: {res_1b:.4f}"
        )

        if best_res_1a is None or res_1a["fcubed"] > best_res_1a["fcubed"]:
            best_res_1a = res_1a
            best_res_1b = res_1b
            best_th = th
            best_preds_1 = preds_1

    print(f"\n[*] Selected optimal threshold: {best_th:.2f} (1a FCubed={best_res_1a['fcubed']:.4f})")

    # 4. Fit and predict Subtask 2
    model_2 = TargetTokenMatchingModel(extractor=extractor)
    model_2.fit(ds.st2_records, language="SV")
    best_preds_2 = model_2.predict(ds.st2_records, language="SV")

    # 5. Output final optimal predictions
    final_st1_file = output_dir / "SV_subtask1.jsonl"
    final_st2_file = output_dir / "SV_subtask2.jsonl"
    write_jsonl(best_preds_1, final_st1_file)
    write_jsonl(best_preds_2, final_st2_file)

    # Also generate other dummy languages for complete test set packaging
    other_files = [final_st1_file, final_st2_file]
    for other_lang in ["IT", "EN"]:
        o_dir = dev_root / other_lang
        if o_dir.exists():
            o_ds = DiachronicDataset.load_from_dir(o_dir, other_lang)
            if o_ds.st1_records:
                p1 = model_1.predict(o_ds.st1_records)
                f1 = output_dir / f"{other_lang}_subtask1.jsonl"
                write_jsonl(p1, f1)
                other_files.append(f1)
            if o_ds.st2_records:
                p2 = model_2.predict(o_ds.st2_records, language=other_lang)
                f2 = output_dir / f"{other_lang}_subtask2.jsonl"
                write_jsonl(p2, f2)
                other_files.append(f2)

    create_submission_zip(submission_zip, other_files)
    print(f"[*] Successfully packaged final target-token submission into {submission_zip}")

    final_eval = evaluator.evaluate(output_dir, languages=["SV"])
    print("")
    print("=" * 60)
    print("🏆 FINAL TARGET-TOKEN MODEL BENCHMARK RESULTS (SV):")
    print("=" * 60)
    print(json.dumps(final_eval, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_target_tuning(
        dev_root=Path("data/raw"),
        output_dir=Path("experiments/target_token_run"),
        submission_zip=Path("submissions/target_token_submission.zip"),
    )
