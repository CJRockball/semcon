# src/semcon/config.py
from pydantic import BaseModel, ConfigDict, Field
import tomllib
from semcon.paths import ROOT

# --- Data decisions (frozen after EDA; see validation.md) ---
CUTOFF = "2008-10-05 05:30:00"   # pd.Timestamp(...) once EDA decides; None = full frame, split='unassigned'
HOLDOUT_FRACTION = None

# Thresholds reproduce the legacy drop_basic() run exactly (590 -> 257);
# provenance: flat-file EDA, 2026-08-26..29 run artifacts.


# --- Column quality rules (EDA-derived; see artifacts/eda_*/eda_summary.md) ---
NAN_FRAC_MAX = 0.5             # drop columns with >50% missing
DOMINANT_FRAC = 0.99           # dominant value in >99% of non-null rows flags NZV
CV_LIMIT = 0.01                # coefficient-of-variation floor
FEW_UNIQUE = 5                 # fewer unique values than this flags NZV
MIN_DEVIANT_N = 5              # deviant rows needed to attempt a rescue
DEVIANT_FAIL_ENRICHMENT = 2.0  # deviants >= 2x base fail rate rescues the column
FEATURE_CORR = 0.95            # correlated pair threshold; drop the more-missing


# --- Engineered feature inputs (EDA-derived; see eda_summary.md) ---
# Legacy 0-based indices converted: legacy n -> s(n+1:03d)
CLIQUE_14 = ["s073", "s074", "s346", "s347"]   # legacy 72,73,345,346; OR 0.49, p=0.0008
CLIQUE_23 = ["s113", "s248", "s386", "s520"]   # legacy 112,247,385,519
BLOCK5_FIRST = 543                             # legacy block=542 (0-based) -> s543..s590


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



