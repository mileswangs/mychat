#!/bin/bash
export OMP_NUM_THREADS=1
export MYCHAT_BASE_DIR="$HOME/.cache/mychat"
mkdir -p $MYCHAT_BASE_DIR

# ---------------------------------------------------------
#python environment and uv 

#intall uv (if not installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# ---------------------------------------------------------
# Wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# ---------------------------------------------------------
# we will be writting markdown report to the report/ directory in the base dir.
# this command clears it and writes a header section
python -m mychat.report reset

# ---------------------------------------------------------
# Tokenizer

# Install rust/cargo 
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
# build the Rustbpe tokenizer
uv run maturin develop --release --mainfest-path rustbpe/Cargo.toml

# Download the first ~2B characters of pretrain dataset
# look at dev/repackage_data_reference.py for details on how the data was prepared
# each data shard is ~250M char
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text(compressed), so this is about 800MB of on disk
python -m mychat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 240 is the right number here
python -m mychat.dataset -n 240 & DATASET_DOWNLOAD_PID=$!
# Train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
python -m scripts.tok_train --max_chars=2000000000 
# evaluate the tokenizer (report compression ratio)
python -m scripts.tok_eval


# ---------------------------------------------------------
# Pretrain

#download the eval bundle to evaluate Core metric when training
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$MYCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $MYCHAT_BASE_DIR
fi

#download identify conversations to impart a personality to the model
if [ ! -f "$MYCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    echo "Downloading identity conversations..."
    curl -L -o "$MYCHAT_BASE_DIR/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi