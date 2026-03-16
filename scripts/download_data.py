import subprocess
import zipfile
from pathlib import Path


DATASET = "avazu-ctr-prediction"


def download_from_kaggle(data_dir: Path):
    """Download dataset using Kaggle CLI."""
    print("Downloading dataset from Kaggle...")

    subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            DATASET,
            "-p",
            str(data_dir),
        ],
        check=True,
    )


def unzip_files(data_dir: Path):
    """Extract downloaded zip files."""
    for zip_file in data_dir.glob("*.zip"):
        print(f"Extracting {zip_file.name}")
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(data_dir)

        zip_file.unlink()


def main():
    project_root = Path(__file__).resolve().parents[1]

    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    download_from_kaggle(raw_dir)
    unzip_files(raw_dir)

    print("Dataset ready in:", raw_dir)


if __name__ == "__main__":
    main()
