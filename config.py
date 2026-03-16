from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Config:
    random_state: int = 42

    # Base directories
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)

    # Files
    model_path: Path = field(init=False)
    pipeline_path: Path = field(init=False)
    submission_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.artifacts_dir = self.project_root / "models"

        self.model_path = self.artifacts_dir / "model.joblib"
        self.pipeline_path = self.artifacts_dir / "pipeline.joblib"
        self.submission_path = self.artifacts_dir / "submission.csv"
