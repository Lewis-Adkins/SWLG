import torch as tc
import numpy as np
from torch.utils.data import Dataset, DataLoader
class M1Dataset(Dataset):
    def __init__(self, data, targets):
        self.data    = data          # (instances, 25, 3)
        self.targets = targets    # (instances,)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = tc.tensor(self.data[idx], dtype=tc.float32)      # already (25, 3)
        y = tc.tensor(self.targets[idx], dtype=tc.float32)   # scalar
        return {"Current_Flux_Data": x, "Target_Proton_Data": y}