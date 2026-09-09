"""Packaging utility to create competition-compliant submission zip files."""

import zipfile
from pathlib import Path
from typing import Dict, List, Sequence
from src.common.logger import get_logger

logger = get_logger("packaging")


def create_submission_zip(
    output_zip_path: Path | str,
    file_paths: Sequence[Path | str],
    allow_overwrite: bool = True,
) -> Path:
    """Packages given prediction jsonl files into a submission zip.

    All files are stored at the root of the zip archive as required by the
    official evaluation protocol.
    """
    output_path = Path(output_zip_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not allow_overwrite:
        raise FileExistsError(f"Submission zip already exists at {output_path}")

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in file_paths:
            path = Path(f)
            if not path.is_file():
                raise FileNotFoundError(f"Cannot package non-existent file: {path}")
            # Ensure file is placed at root of archive
            zf.write(path, arcname=path.name)
            logger.info(f"Added {path.name} to {output_path.name}")

    logger.info(f"Successfully packaged {len(file_paths)} files into {output_path}")
    return output_path
