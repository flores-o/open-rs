#!/bin/bash

set -e  # Stop the script if any command fails

# Step 0: Configure Git identity
if [ -f .env ]; then
    export $(grep -E '^GIT_(NAME|EMAIL)=' .env | xargs)
fi

if [ -z "$GIT_NAME" ] || [ -z "$GIT_EMAIL" ]; then
    echo "⚠️ Please provide your Git identity in a .env file:"
    echo "    GIT_NAME=your-name-here"
    echo "    GIT_EMAIL=your-email@example.com"
    exit 1
fi

if ! git config --global user.email &> /dev/null; then
    echo "🖋️ Setting Git identity..."
    git config --global user.name "$GIT_NAME"
    git config --global user.email "$GIT_EMAIL"
else
    echo "✅ Git identity already configured"
fi

# Step 1: Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Step 2: Install UV if not already installed
if ! command -v uv &> /dev/null; then
    echo "🔧 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "✅ uv already installed"
fi

# Step 2.4: Install Python 3.11 if not already installed
if ! command -v python3.11 &> /dev/null; then
    echo "📦 Installing Python 3.11..."

    # Add deadsnakes PPA for Python 3.11 (Ubuntu/Debian)
    apt-get update
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
    apt-get install -y python3.11 python3.11-venv python3.11-dev

    echo "✅ Python 3.11 installed"
fi


# Step 2.5: Ensure Python 3.11 is installed
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11 is not installed!"
    exit 1
else
    echo "✅ Python 3.11 found!"
fi

# Step 3: Set up virtual environment if it doesn't exist
if [ ! -d "openr1" ]; then
    echo "📦 Creating virtual environment..."
    uv venv openr1 --python 3.11
fi

# Step 4: Activate the environment
source openr1/bin/activate

# Step 5: Upgrade pip and prepare environment
uv pip install --upgrade pip
export UV_LINK_MODE=copy

# Step 6: Install pinned versions for compatibility

echo "🧹 Cleaning any old torch installs..."
pip uninstall -y torch torchvision torchaudio vllm flash-attn || true

CUDA_INDEX=https://download.pytorch.org/whl/cu121

uv pip install \
  torch==2.5.1+cu121 \
  torchvision==0.20.1+cu121 \
  torchaudio==2.5.1+cu121 \
  --extra-index-url "$CUDA_INDEX"



echo "📥 Installing vLLM and FlashAttention..."
uv pip install vllm==0.7.2
uv pip install flash-attn --no-build-isolation
uv pip install setuptools

# Step 7: Install dev dependencies (editable mode)
echo "🔧 Installing development dependencies..."
if ! uv pip install -e ".[dev]"; then
    echo "⚠️ uv pip install failed, falling back to pip..."
    pip install -e ".[dev]"
fi

# Step 8: Ensure Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "🧩 Installing Git LFS..."
    apt-get update && apt-get install -y git-lfs
else
    echo "✅ Git LFS already installed"
fi

# Step 9: Authenticate Hugging Face & Weights & Biases
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

echo "✨  Run: source openr1/bin/activate  (to enter the kingdom)"
