#!/usr/bin/env bash
# Environment setup script for GUI Grounding project
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

ENV_NAME="gui-grounding"

echo "========================================"
echo "GUI Grounding — Environment Setup"
echo "========================================"
echo "Project root: $PROJECT_ROOT"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH."
    echo "Please install Miniconda/Anaconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment '${ENV_NAME}' with Python 3.10..."
    conda create -n "$ENV_NAME" python=3.10 -y
    echo "Conda environment '${ENV_NAME}' created."
else
    echo "Conda environment '${ENV_NAME}' already exists."
fi

# Activate
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "Activated: $(which python) (Python $(python --version 2>&1 | cut -d' ' -f2))"

# Upgrade pip
pip install --upgrade pip

# Install project in editable mode with dev dependencies
echo "Installing project dependencies..."
pip install -e "$PROJECT_ROOT[dev]"

# Create .env from example if it doesn't exist
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    echo "Created .env from .env.example — please fill in your API keys."
fi

# Create data directories
mkdir -p "$PROJECT_ROOT/data"/{raw,interim,processed,manifests}
mkdir -p "$PROJECT_ROOT/outputs"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Activate:  conda activate ${ENV_NAME}"
echo "  2. Edit:      .env (add HF_TOKEN, WANDB_API_KEY, etc.)"
echo "  3. Test:      pytest tests/ -v"
echo "  4. Smoke:     python scripts/run_generate_candidates.py --config configs/train/rerank_reward.yaml"
