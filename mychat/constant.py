import os

BASE_DIR = "./"
DATA_DIR = os.path.join(BASE_DIR, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

base_url = (
    "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
)
max_shard = 1822

index_to_filename = lambda index: f"shard_{index:05d}.parquet"

identity_conversations_filepath = os.path.join(BASE_DIR, "identity_conversations.jsonl")
