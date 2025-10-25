import os
import requests
import time
from mychat.constant import index_to_filename, base_url, max_shard, data_dir


def download_single_file(index: int):
    filename = index_to_filename(index)
    file_path = os.path.join(data_dir, filename)
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
