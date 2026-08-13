import torch as tc
import numpy as np
from torch.utils.data import Dataset, DataLoader
class M1Dataset(Dataset):
    def __init__(self, data, targets):
        # convert once up front instead of per-__getitem__ call, so each
        # epoch only slices existing tensors rather than re-allocating them
        self.data    = tc.as_tensor(data, dtype=tc.float32)      # (instances, 25, 3)
        self.targets = tc.as_tensor(targets, dtype=tc.float32)   # (instances,)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {"Current_Flux_Data": self.data[idx], "Target_Proton_Data": self.targets[idx]}


class GPUBatcher:
    """DataLoader replacement for datasets small enough to live on the GPU
    in full (this project's ~140MB). Moves data to device once at
    construction instead of per-batch, and shuffles/slices with GPU tensor
    indexing instead of a multi-process DataLoader, which only pays off when
    there's real per-sample CPU preprocessing to parallelize."""

    def __init__(self, data, targets, batch_size, device, shuffle=True):
        self.data = tc.as_tensor(data, dtype=tc.float32, device=device)
        self.targets = tc.as_tensor(targets, dtype=tc.float32, device=device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    def __len__(self):
        n = self.data.shape[0]
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = self.data.shape[0]
        idx = tc.randperm(n, device=self.device) if self.shuffle else tc.arange(n, device=self.device)
        for start in range(0, n, self.batch_size):
            batch_idx = idx[start:start + self.batch_size]
            yield {"Current_Flux_Data": self.data[batch_idx], "Target_Proton_Data": self.targets[batch_idx]}