from collections import deque

import torch

from mychat.common import get_dist_info
from mychat.dataset import parquets_iter_batched
from mychat.tokenizer import get_tokenizer


def tokenizing_distributed_data_loader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=1024, device="cuda"):
    """stream pretraining text from parquet files, tokenize, and yield token batches."""
    assert split in ["train", "val"]
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1  # +1 is because we also need the target at the last token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one interation
    token_buffer = deque()

    # infinite iterator over document batches
    def document_batches():
        while True:
            # batch will iterate in group size of the parquet files, usually e.g. 1024 rows
            for batch in parquets_iter_batched(split=split, start=ddp_rank, step=ddp_world_size):
                for i in range(0, len(batch), tokenizer_batch_size):
                    texts = batch[i : i + tokenizer_batch_size]
                    yield texts

    batches = document_batches()

    batch_index = 0
    while True:
        # Accumulate enough tokens for one iteration before yielding
        while len(token_buffer) < needed_tokens:
            doc_path = next(batches)
            token_lists = tokenizer.encode(doc_path, prepend=bos_token, num_threads=tokenizer_threads)
            for token_list in token_lists:
                token_buffer.extend(token_list)
            batch_index += 1
        # Move tokens from the deque into the scatch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # CUDA supports memory pinning for faster transfers between CPU and GPU:
        scratch = torch.tensor(tokens, dtype=torch.int64, pin_memory=(device == "cuda"))
        # Create the input/targets as 1D tensors
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, T).to(device=device, dtype=torch.int32, non_blocking=True)
        targets = targets_cpu.view(B, T).to(device=device, dtype=torch.int64, non_blocking=True)
        yield inputs, targets
