"""
dataset_seediv.py  (v2)
========================
PyTorch Dataset wrapper for preprocessed SEED-IV EEG windows.
Unchanged from v1 except class_weights() is now exposed for use
with the weighted CrossEntropyLoss in the training scripts.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

EMOTION_CLASSES = {0: 'Neutral', 1: 'Sad', 2: 'Fear', 3: 'Happy'}


class SeedIVDataset(Dataset):
    """
    Dataset for SEED-IV preprocessed EEG windows.

    Parameters
    ----------
    data      : np.ndarray  (N, 62, 800)  float32
    label     : np.ndarray  (N,)           int64  {0,1,2,3}
    normalise : bool  Per-sample per-channel z-score at __getitem__ time.
                      Leave False if already done in preprocessing (default).
    transform : callable or None
    """

    def __init__(self, data: np.ndarray, label: np.ndarray,
                 normalise: bool = False, transform=None):
        super().__init__()
        self.data      = data.astype(np.float32)
        self.labels    = label.astype(np.int64)
        self.normalise = normalise
        self.transform = transform
        assert len(self.data) == len(self.labels)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        eeg   = self.data[index]        # (62, 800)
        label = int(self.labels[index])
        if self.normalise:
            eeg = self._ch_zscore(eeg)
        t = torch.from_numpy(eeg)
        if self.transform is not None:
            t = self.transform(t)
        return t, label

    @staticmethod
    def _ch_zscore(eeg: np.ndarray) -> np.ndarray:
        """Per-channel z-score. eeg: (62, 800)"""
        mean = eeg.mean(axis=1, keepdims=True)
        std  = eeg.std(axis=1,  keepdims=True)
        return (eeg - mean) / np.where(std < 1e-8, 1e-8, std)

    def class_counts(self) -> dict:
        return {EMOTION_CLASSES[k]: int((self.labels == k).sum())
                for k in range(4)}

    def class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency weights for CrossEntropyLoss(weight=...).
        Shape: (4,)  FloatTensor.
        """
        counts  = np.array([(self.labels == k).sum() for k in range(4)],
                           dtype=np.float32)
        counts  = np.where(counts == 0, 1.0, counts)
        weights = 1.0 / counts
        weights = weights / weights.sum() * 4.0   # mean = 1
        return torch.from_numpy(weights)

    def summary(self, tag: str = '') -> None:
        prefix = f'[{tag}] ' if tag else ''
        counts = self.class_counts()
        total  = len(self)
        parts  = [f'{k}: {v} ({100*v/total:.1f}%)' for k, v in counts.items()]
        print(f'{prefix}SeedIVDataset  N={total}  shape={self.data.shape}')
        print(f'{prefix}  Class dist → ' + ' | '.join(parts))
