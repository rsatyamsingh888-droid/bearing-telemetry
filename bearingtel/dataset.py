import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DATA = Path("data/processed")
CLASSES = ["healthy", "inner", "outer", "ball", "cage"]


class WindowDataset(Dataset):
    def __init__(self, split):
        arrays = np.load(DATA / f"{split}.npz")
        norm = json.loads((DATA / "norm.json").read_text())
        mean = np.array(norm["mean"], dtype=np.float32).reshape(1, -1, 1)
        std = np.array(norm["std"], dtype=np.float32).reshape(1, -1, 1)


        self.x = torch.from_numpy((arrays["x"] - mean) / std)
        self.y = torch.from_numpy(arrays["y"]).long()
        self.bearing = arrays["bearing"]
        self.state = arrays["state"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


if __name__ == "__main__":
    ds = WindowDataset("train")
    x, y = ds[0]
    print("windows:", len(ds))
    print("x:", x.shape, x.dtype, f"mean {ds.x.mean():.3f} std {ds.x.std(dim=(0, 2))}")
    print("y:", y, CLASSES[y])        

