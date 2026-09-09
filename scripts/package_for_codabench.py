"""Package submission tailored for Codabench online evaluation environment."""

import argparse
import json
from pathlib import Path
from src.common.packaging import create_submission_zip


def package_online(
    input_dir: Path = Path("experiments/production_champion"),
    output_zip: Path = Path("submissions/online_ready_submission.zip"),
    target_words: list[str] | None = None,
):
    input_dir = Path(input_dir)
    output_tmp = input_dir / "_online_filtered"
    output_tmp.mkdir(parents=True, exist_ok=True)

    target_words_set = set(target_words) if target_words else None
    print(f"[*] Packaging for Codabench with target words filter: {target_words_set}")

    packaged_files = []
    for lang in ["SV"]:
        for st in ["subtask1", "subtask2"]:
            src_file = input_dir / f"{lang}_{st}.jsonl"
            dest_file = output_tmp / f"{lang}_{st}.jsonl"
            if not src_file.exists():
                continue

            count = 0
            with open(src_file, "r", encoding="utf-8") as fin, open(dest_file, "w", encoding="utf-8") as fout:
                for line in fin:
                    rec = json.loads(line)
                    if target_words_set is None or rec.get("word") in target_words_set:
                        fout.write(line)
                        count += 1

            packaged_files.append(dest_file)
            print(f"    -> Filtered {dest_file.name}: {count} records")

    create_submission_zip(output_zip, packaged_files)
    print(f"\n[+] Successfully packaged online-ready zip: {output_zip}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--words", nargs="*", default=["kvinna", "motiv"], help="Filter words (default: kvinna motiv for current Codabench server state)")
    parser.add_argument("--output-zip", type=Path, default=Path("submissions/online_ready_submission.zip"))
    args = parser.parse_args()

    package_online(
        input_dir=Path("experiments/production_champion"),
        output_zip=args.output_zip,
        target_words=args.words,
    )
