"""Project-wide constants for GUI grounding."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MANIFESTS_DIR = DATA_DIR / "manifests"
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Action types supported by the grounding task
# ---------------------------------------------------------------------------
ACTION_TYPES = ("click", "type", "select", "hover")
ACTION_TYPE_TO_ID = {a: i for i, a in enumerate(ACTION_TYPES)}
ID_TO_ACTION_TYPE = {i: a for a, i in ACTION_TYPE_TO_ID.items()}
NUM_ACTION_TYPES = len(ACTION_TYPES)

# ---------------------------------------------------------------------------
# Dataset identifiers
# ---------------------------------------------------------------------------
DATASET_NAMES = (
    "mind2web",
    "screenspot_v2",
    "visualwebbench",
    "screenspot_pro",
)

DATASET_ROLES = {
    "mind2web": "train+eval",
    "screenspot_v2": "eval",
    "visualwebbench": "supplementary_eval",
    "screenspot_pro": "optional_hard_eval",
}

# ---------------------------------------------------------------------------
# Splits for Mind2Web generalization study
# ---------------------------------------------------------------------------
MIND2WEB_SPLITS = ("train", "test_task", "test_website", "test_domain")

# ---------------------------------------------------------------------------
# Default reward weights (λ1 … λ5)
# ---------------------------------------------------------------------------
DEFAULT_REWARD_WEIGHTS = {
    "element_correct": 1.0,
    "iou": 0.5,
    "click_inside_target": 0.3,
    "action_type_correct": 0.2,
    "invalid_format_penalty": 0.5,
}

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
SUPPORTED_BACKBONES = (
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
)
