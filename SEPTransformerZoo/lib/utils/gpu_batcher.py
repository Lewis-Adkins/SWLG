from collections.abc import Iterator

import numpy as np
import torch


class GPUBatcher:
    def __init__(self, data: np.ndarray, targets: np.ndarray, batch_size: int, device: torch.device,
                 shuffle: bool = True, drop_last: bool = False):
        self.data       = torch.as_tensor(data, dtype=torch.float32, device=device)
        self.targets    = torch.as_tensor(targets, dtype=torch.float32, device=device)
        self.batch_size = batch_size
        self.device     = device
        self.shuffle    = shuffle
        self.drop_last  = drop_last

    def __len__(self) -> int:
        n = self.data.shape[0]
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[dict]:
        n = self.data.shape[0]
        if self.drop_last:
            n = (n // self.batch_size) * self.batch_size
        idx = torch.randperm(n, device=self.device) if self.shuffle else torch.arange(n, device=self.device)
        for start in range(0, n, self.batch_size):
            batch_idx = idx[start:start + self.batch_size]
            yield {"Current_Flux_Data": self.data[batch_idx], "Target_Proton_Data": self.targets[batch_idx]}
