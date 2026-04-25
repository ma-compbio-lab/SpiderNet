from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
from .utils import get_default_cellchat_db, get_default_scseqcomm_db

@dataclass
class PreprocessConfig:
    n_hvg: int = 1000
    n_hvg_lr: int = 2000
    num_neighbors: int = 10

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class TrainingConfig:
    version: str = "V1"
    dim_envir: int = 15
    max_epoch: int = 20000
    n_jobs: int = 5
    optimizer: str = "adam"

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class PathConfig:
    data_root: Path
    output_root: Path
    species: str = "human"
    cellchat_db: Optional[Path] = None
    scseqcomm_db: Optional[Path] = None
    version: str = "V1"

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.output_root = Path(self.output_root)

        if self.cellchat_db is None:
            self.cellchat_db = get_default_cellchat_db(self.species)
        else:
            self.cellchat_db = Path(self.cellchat_db)

        if self.scseqcomm_db is None:
            self.scseqcomm_db = get_default_scseqcomm_db(self.species)
        else:
            self.scseqcomm_db = Path(self.scseqcomm_db)

    def resolve_run_dir(self, dim_envir: int) -> Path:
        result_tag = f"SpiderNet_Result_dim{dim_envir}"
        return self.output_root / self.version / result_tag

    def ensure_dirs(self, dim_envir: int) -> dict[str, Path]:
        run_dir = self.resolve_run_dir(
            dim_envir=dim_envir
        )
        model_dir = run_dir / "Model"

        for path in (run_dir, model_dir):
            path.mkdir(parents=True, exist_ok=True)

        return {
            "run_dir": run_dir,
            "model_dir": model_dir,
        }
