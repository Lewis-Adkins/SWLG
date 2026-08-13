from sklearn.utils import resample
import numpy as np
from scipy.stats import bootstrap
import pandas as pd
n_iter = 10
x = np.random.default_rng().random(10)
X = np.array([])
n_samples  = int(len(x))
rs = 42
data = pd.read_csv('data/data.csv')

proton = data['proton'].values
e_h = data['electron_high'].values
e_l = data['electron'].values

data = np.array([proton, e_h, e_l])
size_blocks = 3

def moving_block_data(data, size_blocks, n_datasets):

    blocks = []
     
    n_blocks = data.shape[1] - size_blocks + 1
    print(type(data))
    for i in range(n_blocks):
        block = data[:, i : i + size_blocks]
        blocks.append(block)

    key_blocks = np.arange(n_blocks)
    n_blocks_needed = -(-data.shape[1] // size_blocks)  # ceil

    master_data = []
    for j in range(n_datasets):
        samples_indices = resample(key_blocks, replace = True, n_samples= n_blocks_needed, random_state=42 + j )

        new_dataset = []

        for si in samples_indices:
            new_dataset.append(blocks[si])

        new_dataset = np.hstack(new_dataset)
        master_data.append(new_dataset)    

    return master_data
print(moving_block_data(data, 1000,10))
