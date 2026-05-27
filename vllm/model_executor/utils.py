"""Utils for model executor."""
import random

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# flexllm
def tensor_copy(a: torch.Tensor, b: torch.Tensor):
    assert len(a.shape) == len(b.shape)
    dim = len(a.shape)
    slice_indices = [slice(None)] * dim

    for i in range(dim):
        index_a = a.shape[i]
        index_b = b.shape[i]
        index = min(index_a, index_b)
        slice_indices[i] = slice(0, index)

    a.zero_()
    a[slice_indices].copy_(b[slice_indices])
