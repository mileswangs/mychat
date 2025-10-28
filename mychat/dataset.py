import argparse
import os
import requests
import time
import pyarrow.parquet as pq
from multiprocessing import Pool

from mychat.common import get_base_dir()

DATA_DIR = os.path.join(get_base_dir(), "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

base_url = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
max_shard = 1822

index_to_filename = lambda index: f"shard_{index:05d}.parquet"

identity_conversations_filepath = os.path.join(get_base_dir(), "identity_conversations.jsonl")


def list_parquet_files(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet")])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"]
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            text = rg.column("text").to_pylist()
            yield text


def download_single_file(index: int):
    filename = index_to_filename(index)
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        print(f"{file_path} already exists, skipping download.")
        return
    url = f"{base_url}/{filename}"
    print(f"Downloading {filename}")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            temp_path = file_path + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, file_path)
            print(f"Downloaded {filename} successfully.")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # cleanup temp file if exists
            for file in [temp_path, file_path]:
                if os.path.exists(file):
                    os.remove(file)
            if attempt < max_attempts:
                wait_time = 2**attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts.")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument(
        "-n",
        "--num-files",
        type=int,
        default=-1,
        help="Number of files to download (default= -1, -1 = disable)",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads to use (default=4)",
    )
    args = parser.parse_args()

    num = max_shard + 1 if args.num_files == -1 else min(args.num_files, max_shard + 1)
    ids_to_download = list(range(num))
    print(f"Downloading of {len(ids_to_download)} files using {args.num_workers} workers.")
    print(f"target directory: {DATA_DIR}")
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    successful_downloads = sum(1 for result in results if result)
    print(f"Downloaded {successful_downloads}/{len(ids_to_download)} files successfully.")
