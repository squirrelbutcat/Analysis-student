from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub


DATASET = "shivasingh4945/student-social-media-and-mental-health-impact"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_path = Path(kagglehub.dataset_download(DATASET))
    print(f"Downloaded dataset to cache: {downloaded_path}")

    copied = []
    for file_path in downloaded_path.rglob("*"):
        if file_path.is_file():
            target = RAW_DIR / file_path.name
            shutil.copy2(file_path, target)
            copied.append(target)

    if not copied:
        raise FileNotFoundError(f"No files found in downloaded dataset: {downloaded_path}")

    print("Copied files to data/raw:")
    for path in copied:
        print(f"- {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
