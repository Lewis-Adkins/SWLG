import numpy as np
import torch
from torch.utils.data import Dataset


class ChampionDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray):
        self.data    = torch.as_tensor(data, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {"Current_Flux_Data": self.data[idx], "Target_Proton_Data": self.targets[idx]}
