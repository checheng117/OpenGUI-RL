#!/usr/bin/env bash
# Python 3.10-first environment setup for Qwen-VL realigned path.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_NAME="${ENV_NAME:-gui-grounding-py310}"
USE_CONDA="${USE_CONDA:-auto}"   # auto|1|0
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv-py310}"

echo "========================================"
echo "GUI Grounding — Py3.10 Environment Setup"
echo "========================================"
echo "Project root: $PROJECT_ROOT"

activate_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "Creating venv at $VENV_DIR with python3.10..."
        python3.10 -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
}

if [[ "$USE_CONDA" == "1" || "$USE_CONDA" == "auto" ]]; then
    if command -v conda >/dev/null 2>&1; then
        echo "Using conda environment '$ENV_NAME' (python=3.10)..."
        export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-}"
        eval "$(conda shell.bash hook)"
        if ! conda env list | grep -Eq "^${ENV_NAME}[[:space:]]"; then
            conda create -n "$ENV_NAME" python=3.10 -y
        fi
        # Some conda activate hooks reference unset vars or fail on strict mode.
        set +e
        conda activate "$ENV_NAME"
        act_status=$?
        set -e
        if [[ $act_status -ne 0 ]]; then
            echo "Conda activation failed; falling back to local venv."
            activate_venv
        fi
    elif [[ "$USE_CONDA" == "1" ]]; then
        echo "Error: USE_CONDA=1 but conda not found in PATH."
        exit 1
    else
        echo "Conda not found, falling back to local venv."
        activate_venv
    fi
else
    activate_venv
fi

PY_VER="$(python --version 2>&1 | awk '{print $2}')"
if [[ "$PY_VER" != 3.10.* ]]; then
    echo "Error: active interpreter is Python $PY_VER (expected 3.10.x)."
    echo "Please install Python 3.10 and rerun this script."
    exit 1
fi
echo "Activated Python: $(which python) ($PY_VER)"

echo "Installing core dependencies for Qwen-VL path..."
python -m pip install --upgrade pip wheel setuptools

# Install a CUDA stack compatible with RTX 3090 + current 12.x drivers.
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Installing PyTorch CUDA wheel set (cu124)..."
    python -m pip install \
        "torch==2.5.1" \
        "torchvision==0.20.1" \
        "torchaudio==2.5.1" \
        --index-url "https://download.pytorch.org/whl/cu124"
else
    echo "No NVIDIA GPU runtime found; installing CPU PyTorch wheels."
    python -m pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"
fi

# Base project deps + dev tooling used by scripts/tests.
python -m pip install -e "$PROJECT_ROOT[dev]"

# Explicit runtime deps for real Qwen-VL backbones.
python -m pip install \
    "transformers>=4.51.0" \
    "accelerate>=0.30.0" \
    "safetensors>=0.4.3" \
    "huggingface_hub>=0.23.0" \
    "pillow>=10.0.0" \
    "torchvision>=0.16.0" \
    "qwen-vl-utils>=0.0.8" \
    "python-dotenv>=1.0.1"

if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    echo "Created .env from .env.example"
fi

mkdir -p "$PROJECT_ROOT/data"/{raw,interim,processed,manifests}
mkdir -p "$PROJECT_ROOT/outputs"

echo ""
echo "========================================"
echo "Setup complete (Py3.10, Qwen-primary path)"
echo "========================================"
echo ""
echo "Activation:"
if command -v conda >/dev/null 2>&1 && [[ "$USE_CONDA" != "0" ]]; then
    echo "  conda activate $ENV_NAME"
else
    echo "  source \"$VENV_DIR/bin/activate\""
fi
echo ""
echo "Recommended smoke commands:"
echo "  python scripts/run_single_inference.py --config configs/demo/single_inference.yaml --dry-run"
echo "  python scripts/run_train_sft.py --config configs/train/sft_qwen2_5_vl_3b_stagea.yaml --dry-run"
