#!/bin/bash

echo "🌟 Starting Open-RS environment setup..."

# Step 1: Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Step 2: Install UV if not already installed
if ! command -v uv &> /dev/null; then
    echo "🔧 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.local/bin/env
else
    echo "✅ uv already installed"
fi

# Step 3: Set up the virtual environment if it doesn't exist
if [ ! -d "openr1" ]; then
    echo "📦 Creating virtual environment..."
    uv venv openr1 --python 3.11
fi

# Step 4: Activate the environment
source openr1/bin/activate

# Step 5: Upgrade pip and prepare environment
uv pip install --upgrade pip
export UV_LINK_MODE=copy

# Step 6: Install core dependencies (skip if already installed)
if ! pip show vllm &> /dev/null; then
    echo "🧠 Installing vLLM, setuptools, and flash-attn..."
    uv pip install vllm==0.7.2
    uv pip install setuptools
    uv pip install flash-attn --no-build-isolation
else
    echo "✅ Core dependencies already installed"
fi

# Step 7: Install dev dependencies (editable mode)
echo "🔧 Installing development dependencies..."
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"

# Step 8: Ensure Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "🧩 Installing Git LFS..."
    apt-get update && apt-get install -y git-lfs
else
    echo "✅ Git LFS already installed"
fi

# Step 9: Authenticate with Hugging Face & Weights & Biases
if [ -f .env ]; then
    echo "🔐 Loading credentials from .env..."

    HF_TOKEN=$(grep -E '^HUGGINGFACE_TOKEN=' .env | cut -d '=' -f2-)
    WANDB_API_KEY=$(grep -E '^WANDB_API_KEY=' .env | cut -d '=' -f2-)

    if [ -n "$HF_TOKEN" ]; then
        huggingface-cli login --token "$HF_TOKEN"
    else
        echo "⚠️ HUGGINGFACE_TOKEN not set in .env"
    fi

    if [ -n "$WANDB_API_KEY" ]; then
        wandb login --relogin "$WANDB_API_KEY"
    else
        echo "⚠️ WANDB_API_KEY not set in .env"
    fi
else
    echo "⚠️ No .env file found — please copy .env.template and fill it in"
fi

echo "🎉 All done! Environment is ready to reign. 👑"
