import os

base_dir = "./"
data_dir = os.path.join(base_dir, "base_data")
os.makedirs(data_dir, exist_ok=True)

base_url = (
    "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
)
max_shard = 1822

index_to_filename = lambda index: f"shard_{index:05d}.parquet"
