"""Production pipeline for SemEval-2027 Task 3: multi-language factory, dual ensemble, and packaging."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from src.common.config import ALLOWED_LANGUAGES, RAW_DATA_DIR, SUBMISSIONS_DIR
from src.common.dataset import DiachronicDataset
from src.common.factory import MultilingualModelFactory
from src.common.io import write_jsonl
from src.common.packaging import create_submission_zip
from src.subtask1.target_clustering import TargetTokenClusteringModel
from src.subtask2.calibrator import ConfidenceMarginCalibrator
from src.subtask2.target_matching import TargetTokenMatchingModel
from src.validation.evaluator import LocalEvaluator


def run_production(
    data_root: Path = RAW_DATA_DIR,
    output_dir: Path = Path("experiments/production_run"),
    submission_zip: Path = Path("submissions/champion_submission.zip"),
    use_ensemble: bool = True,
    target_languages: list[str] | None = None,
):
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_zip = Path(submission_zip)

    available_langs = [
        d.name.upper()
        for d in data_root.iterdir()
        if d.is_dir() and d.name.upper() in ALLOWED_LANGUAGES
    ]
    if target_languages:
        available_langs = [l for l in available_langs if l in [t.upper() for t in target_languages]]
    else:
        # If not specified, default to SV for current development phase
        available_langs = [l for l in available_langs if l == "SV"]

    print(f"[*] Target competition languages to process: {available_langs}")

    generated_files = []
    calibrator = ConfidenceMarginCalibrator(temperature=1.2, margin=0.03)

    for lang in available_langs:
        lang_dir = data_root / lang
        ds = DiachronicDataset.load_from_dir(lang_dir, lang)

        if not ds.st1_records and not ds.st2_records:
            continue

        print(f"\n[*] Processing language [{lang}] with MultilingualModelFactory...")
        extractor = MultilingualModelFactory.get_extractor(
            language=lang,
            use_ensemble=use_ensemble,
        )

        # 1. Subtask 1
        if ds.st1_records:
            st1_model = TargetTokenClusteringModel(
                extractor=extractor,
                distance_threshold=0.36,
                min_cluster_size=2,
                linkage="average",
            )
            preds_1 = st1_model.predict(ds.st1_records)
            st1_path = output_dir / f"{lang}_subtask1.jsonl"
            write_jsonl(preds_1, st1_path)
            generated_files.append(st1_path)
            print(f"    -> Generated {st1_path.name} ({len(preds_1)} predictions)")

        # 2. Subtask 2
        if ds.st2_records:
            st2_model = TargetTokenMatchingModel(extractor=extractor)
            st2_model.fit(ds.st2_records, language=lang)
            preds_2 = st2_model.predict(ds.st2_records, language=lang)
            st2_path = output_dir / f"{lang}_subtask2.jsonl"
            write_jsonl(preds_2, st2_path)
            generated_files.append(st2_path)
            print(f"    -> Generated {st2_path.name} ({len(preds_2)} predictions)")

    # Only package languages that actually exist in the benchmark to avoid missing gold errors
    create_submission_zip(submission_zip, generated_files)
    print(f"\n[*] Champion submission packaged: {submission_zip}")

    evaluator = LocalEvaluator(gold_dir=data_root)
    eval_res = evaluator.evaluate(output_dir, languages=available_langs)

    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_res, f, indent=2, ensure_ascii=False)

    print("")
    print("=" * 60)
    print(f"🏆 CHAMPION SUBMISSION VALIDATION REPORT ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("=" * 60)
    print(json.dumps(eval_res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full production pipeline for SemEval-2027 Task 3.")
    parser.add_argument("--data-root", type=Path, default=RAW_DATA_DIR, help="Path to raw data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/production_run"), help="Output work directory")
    parser.add_argument("--submission-zip", type=Path, default=Path("submissions/champion_submission.zip"), help="Output submission zip")
    parser.add_argument("--languages", nargs="*", default=None, help="Target languages to process (e.g. SV)")
    parser.add_argument("--no-ensemble", action="store_true", help="Disable dual-model ensemble")
    args = parser.parse_args()

    run_production(
        data_root=args.data_root,
        output_dir=args.output_dir,
        submission_zip=args.submission_zip,
        use_ensemble=not args.no_ensemble,
        target_languages=args.languages,
    )
