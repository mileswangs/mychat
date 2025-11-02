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
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download minimal data for testing (~1B characters, 4 shards)
# Each shard is ~250M char, so 4 shards = ~1B characters
# This is about 400MB of compressed text on disk
python -m mychat.dataset -n 4
# Train the tokenizer with vocab size 2**16 = 65536 on ~1B characters of data
python -m scripts.tok_train --max_chars=1000000000 
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

# Train a small d4 model for testing (instead of d20)
# This is much faster and sufficient for verifying the pipeline works
# Using 100 iterations instead of full training
# With 8 GPUs, batch_size=1, seq_len=1024: world_tokens = 8*1*1024 = 8192
# So total_batch_size must be a multiple of 8192
# eval_tokens must be >= world_tokens (8192) to have at least 1 eval step
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=4 \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --total_batch_size=8192 \
    --eval_every=50 \
    --eval_tokens=16384 \
    --core_metric_every=50 \
    --core_metric_max_per_task=12 \
    --sample_every=50 \
    --num_iterations=100 \
    --run=$WANDB_RUN
# evaluate the model on a smaller chunk of train/val data
# split_tokens must be >= world_tokens (8192)
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss -- --device_batch_size=1 --split_tokens=16384
# evaluate the model on CORE tasks (reduced problem set for testing)
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --max-per-task=16

# # ------------------------------------------------------------------------------------
# # Midtrain (teach the model conversation special tokens, tool use, multiple turns)
# # Using reduced iterations for testing

#download identify conversations to impart a personality to the model
#see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
if [ ! -f "$MYCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    echo "Downloading identity conversations..."
    curl -L -o "$MYCHAT_BASE_DIR/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

# run midtrain with reduced iterations and eval the model
# With 8 GPUs, batch_size=1, seq_len=1024: world_tokens = 8192
# eval_tokens must be >= 8192
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --eval_every=50 \
    --eval_tokens=16384 \
    --total_batch_size=8192 \
    --num_iterations=100 \
    --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid --max-new-tokens=128 --max-problems=20

# ------------------------------------------------------------------------------------
# Supervised Fine-Tuning (SFT) with reduced iterations for testing

# train sft with reduced iterations and re-eval
# With 8 GPUs and device_batch_size=1: examples_per_step = 1 * 8 = 8
# target_examples_per_step must be a multiple of 8
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --device_batch_size=1 \
    --target_examples_per_step=8 \
    --num_iterations=100 \
    --eval_steps=4 \
    --eval_metrics_max_problems=16 \
    --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft --max-new-tokens=128 --max-problems=20

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# ------------------------------------------------------------------------------------
# Reinforcement learning, currently only on gsm8k (with reduced iterations)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- \
    --num_iterations=50 \
    --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K --max-problems=20

# ------------------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m mychat.report generate
