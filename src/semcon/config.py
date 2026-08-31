# src/semcon/config.py
from pydantic import BaseModel, ConfigDict, Field
import tomllib
from semcon.paths import ROOT

class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kfolds: int = Field(default=5, ge=2)
    tail_n: int = 27
    holdout_n: int = 231
    seed: int = 1337

class SelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    g_min: float = Field(default=0.25, gt=0)
    k_min: int = 30
    k_max: int = 100

class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objective: str = "binary:logistic"
    learning_rate: float = 0.03
    max_depth: int = 4
    max_bin: int = 511
    reg_alpha: float = 3.0
    reg_lambda: float = 2.0
    n_estimators: int = 5000
    

class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: ModelConfig = ModelConfig()
    pipeline: PipelineConfig = PipelineConfig()
    selection: SelectionConfig = SelectionConfig()
    
    def with_model_overrides(self, overrides: dict) -> "Config":
        """Return self with --set overrides applied and re-validated."""
        if overrides:
            self.model = ModelConfig(**{**self.model.model_dump(), **overrides})
        return self


def load_config(path=None) -> Config:
    path = path or ROOT / "config" / "default.toml"
    data = tomllib.load(open(path, "rb")) if path.exists() else {}
    return Config(**data)      # file overrides the model defaults


def parse_overrides(overrides: list[str]) -> dict:
    """Split --set key=value strings. Coercion and validation happen
    when the dict is applied to the config model."""
    out = {}
    for item in overrides:
        key, sep, value = item.partition('=')
        if not sep:
            raise ValueError(f"--set expects key=value, got {item!r}")
        out[key] = value
    return out


# --- Data decisions (frozen after EDA; see validation.md) ---
CUTOFF = None   # pd.Timestamp(...) once EDA decides; None = full frame, split='unassigned'
HOLDOUT_FRACTION = None
