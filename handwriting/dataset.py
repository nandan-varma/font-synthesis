import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config


class HandwritingDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", val_fraction: float = 0.05):
        x     = np.load(f"{data_dir}/x.npy",     mmap_mode='r')
        x_len = np.load(f"{data_dir}/x_len.npy", mmap_mode='r')
        c     = np.load(f"{data_dir}/c.npy",      mmap_mode='r')
        c_len = np.load(f"{data_dir}/c_len.npy",  mmap_mode='r')

        n = len(x)
        split_idx = int(n * (1 - val_fraction))
        sl = slice(None, split_idx) if split == "train" else slice(split_idx, None)

        self.x     = x[sl]
        self.x_len = x_len[sl]
        self.c     = c[sl]
        self.c_len = c_len[sl]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict:
        return {
            "x":     torch.from_numpy(self.x[idx].copy()).float(),
            "x_len": int(self.x_len[idx]),
            "c":     torch.from_numpy(self.c[idx].astype(np.int64).copy()),
            "c_len": int(self.c_len[idx]),
        }


def get_dataloader(cfg: Config, split: str = "train") -> DataLoader:
    ds = HandwritingDataset(cfg.data_dir, split=split)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        persistent_workers=(cfg.num_workers > 0),
        pin_memory=False,    # MPS doesn't support pin_memory
        drop_last=(split == "train"),
    )
