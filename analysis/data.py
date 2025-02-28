from functools import reduce
from ieeg.calc.fast import mixup

import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from ieeg.calc.oversample import MinimumNaNSplit
import torch
import numpy as np
from ieeg.arrays.label import LabeledArray
from ieeg.decoding.decode import nan_common_denom

device = ('cuda' if torch.cuda.is_available() else 'cpu')
# %% Define data module

class LabeledData(L.LightningDataModule):
    def __init__(self, data: LabeledArray, n_folds: int, val_size: float,
                 target_map: dict, target_dim: int = 1):
        super(LabeledData, self).__init__()
        self.data = torch.from_numpy(
            data / np.nanstd(data)).to(device, dtype=torch.float32).permute(1, 2, 0)
        self.labels = data.labels
        targets = []
        for lab in data.labels[target_dim]:
            for key, value in target_map.items():
                if key in lab:
                    targets.append(value)
                    break
        self.targets = torch.as_tensor(targets)
        self.val_size = val_size
        self.folds = n_folds
        self.current_fold = None
        self._fold_idx = None
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

    def set_fold(self, fold):
        assert 0 <= fold < self.folds, "Fold index out of range"
        self.current_fold = fold
        self.train_idx = self._fold_idx[fold]["train"]
        self.val_idx = self._fold_idx[fold]["val"]
        self.test_idx = self._fold_idx[fold]["test"]

    def setup(self, stage=None):
        cv = MinimumNaNSplit(self.folds, 1, which='test')
        split = cv.split(self.data.cpu(), self.targets)
        self._fold_idx = {}
        for i, (train_idx, test_idx) in enumerate(split):
            val_idx = train_idx[torch.distributions.uniform.Uniform(
                0, 1).sample((train_idx.shape[0],)) < self.val_size]
            train_idx = np.setdiff1d(train_idx, val_idx)
            self._fold_idx[i] = {"train": train_idx, "val": val_idx, "test": test_idx}

        self.set_fold(0)

    def train_dataloader(self):
        return DataLoader(TensorDataset(self.data[self.train_idx],
                                        self.targets[self.train_idx]))

    def val_dataloader(self):
        return DataLoader(TensorDataset(self.data[self.val_idx],
                                        self.targets[self.val_idx]))

    def test_dataloader(self):
        return DataLoader(TensorDataset(self.data[self.test_idx],
                                        self.targets[self.test_idx]))


def dataloader(array, idx, conds, metric='zscore', do_mixup=False, no_nan=False):
    array = array[metric, :, :, idx][conds,].dropna()
    ax = array.ndim - 2
    if no_nan:
        nan_common_denom(array, True, ax,  10, 1, True)
    std = np.nanstd(array.__array__())
    if do_mixup:
        mixup(array[metric], ax)
    combined = reduce(lambda x, y: x.concatenate(y, -1),
                      [array[c] for c in conds])
    data = combined.combine((0, 2)).swapaxes(0, 1)
    neural_data_tensor = torch.from_numpy(
        (data.__array__() / std))
    return neural_data_tensor, data.labels
