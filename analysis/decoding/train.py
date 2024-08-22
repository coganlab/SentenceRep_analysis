# %% Imports
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
from ieeg.calc.oversample import MinimumNaNSplit
import torch
import numpy as np
from analysis.decoding.models import CNNTransformer, SimpleDecoder
from analysis.grouping import GroupData
import os
from ieeg.calc.mat import Labels, LabeledArray


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

    def setup(self, stage=None, rep=1):
        cv = MinimumNaNSplit(self.folds, rep, which='test')
        split = cv.split(self.data, self.targets)
        while rep > 1:
            for _ in range(self.folds):
                next(split)
            rep -= 1
        self._fold_idx = {}
        for i, (train_idx, test_idx) in enumerate(split):
            train_idx = train_idx
            val_idx = train_idx[torch.distributions.uniform.Uniform(
                0, 1).sample((train_idx.shape[0],)) < self.val_size]
            test_idx = test_idx
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


# dummy data
n_samples = 144
n_timepoints = 200
n_features = 111
fs = 200

# create the data module
batch_size = 32
n_folds = 5
val_size = 0.2
fpath = os.path.expanduser("~/Box/CoganLab")
# # Create a gridspec instance with 3 rows and 3 columns
device = ('cuda' if torch.cuda.is_available() else 'cpu')
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats')
idx = sorted(list(sub.SM))
aud_slice = slice(0, 175)
reduced = sub[:, :, :, idx][:, ['aud_ls', 'aud_lm', 'aud_jl']]
# reduced = sub[:, :, :, idx][:, ['go_ls', 'go_lm']]

reduced = reduced.nan_common_denom(True, 10, True)
aud = reduced.array['zscore']

idx = [i for i, l in enumerate(sub.array.labels[3]) if
 l in reduced.array.labels[2]]
# transfer data to torch tensor
aud.labels[0] = Labels(aud.labels[0].replace("aud_", ""))
aud = LabeledArray(np.ascontiguousarray(aud.__array__()), aud.labels)
data = aud.combine((0,1)).combine((0,2)).dropna()
# aud_go = LabeledArray(np.ascontiguousarray(aud_go.__array__()), aud_go.labels)
# data = aud_go.combine((0, 1)).combine((0, 2)).__array__()
# del sub

stitched = np.hstack([sub.signif['aud_ls', :, aud_slice],
                      # sub.signif['aud_lm', :, aud_slice],
                      sub.signif['resp', :]])

neural_data_tensor = torch.from_numpy(
    data / np.nanstd(data)).to(device, dtype=torch.float32).permute(1, 2, 0)

# %% Create model
# model parameters
in_channels = data.shape[0]
# num_classes = 4
d_model = 12
kernel_time = 50  # ms
kernel_size = int(kernel_time * fs / 1000)  # kernel length in samples
stride_time = 10 # ms
stride = int(stride_time * fs / 1000)  # stride length in samples
padding = 0
n_head = 4
num_layers = 2
dim_fc = 24
dropout = 0.3
# learning_rate = 1e-5

# instantiate the model
# model = CNNTransformer(in_channels, num_classes, data.shape[1], kernel_size, stride, padding,
#                           n_head, num_layers, dim_fc, dropout, learning_rate)

# %% Train the model with kfold
# instantiate the trainer
n_folds = 5
max_epochs = 500

callbacks = [
             EarlyStopping(monitor='val_loss', patience=3, mode='min',min_delta=0.0001
                           )]
trainer = L.Trainer(max_epochs=max_epochs,
                    accelerator='cpu',
                    callbacks=callbacks,
                    logger=True,
                    )
target_map = {'heat': 0, 'hut': 1, 'hot': 2, 'hoot': 3}


# %% train the model
def process_data(data, n_iters, n_folds, val_size, d_model,
                 target_map, max_epochs, learning_rate = 1e-4, verbose=False):

    dm = LabeledData(data, n_folds, val_size, target_map)

    iter_accs = []
    es_pat = max_epochs // 40
    num_classes = len(target_map)
    for i in range(n_iters):
        dm.setup(rep=i + 1)

        fold_accs = []
        for fold in range(n_folds):
            dm.set_fold(fold)
            # print(dm.current_fold)

            # instantiate the model
            # in_channels = dm.get_data_shape()[-1]
            # model = SimpleDecoder(num_classes, d_model, learning_rate)
            model = CNNTransformer(in_channels, num_classes, d_model, kernel_size, stride, padding,
                                      n_head, num_layers, dim_fc, dropout, learning_rate)
            # model.current_fold = fold
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=es_pat, mode='min', min_delta=0.01
                              )]
            trainer = L.Trainer(max_epochs=max_epochs,
                                # gradient_clip_val=gclip_val,
                                accelerator='auto',
                                callbacks=callbacks,
                                logger=False,
                                enable_model_summary=verbose,
                                enable_progress_bar=verbose,
                                enable_checkpointing=False
                                )
            trainer.fit(model, dm)
            # print(trainer.logged_metrics)
            trainer.test(model, dm, verbose=False)
            fold_accs.append(trainer.logged_metrics['test_acc'])

            # save loss information
            # loss_dict = trainer.logger.metrics
            # loss_dict['fold'] = fold
            # loss_dict['model'] = model
        # print(f'Averaged accuracy: {sum(fold_accs) / len(fold_accs)}')
        iter_accs.append(fold_accs)
    # print(sum(iter_accs) / len(iter_accs), iter_accs)
    # print(iter_accs)
    return torch.as_tensor(iter_accs)

# out = process_data(data, 1, n_folds, val_size, d_model, target_map, max_epochs, verbose=True)

# %% windowed decoding
from analysis.decoding import windower
from joblib import Parallel, delayed

data_windowed = LabeledArray(windower(data, 20, 2).swapaxes(0, -1))
data_windowed.labels[1] = data.labels[0]
data_windowed.labels[2] = data.labels[1]

out = Parallel(n_jobs=4, verbose=40)(delayed(process_data)(
    d, 1, n_folds, val_size, target_map, max_epochs) for d in data_windowed)

# %% plot the results
from ieeg.viz.ensemble import plot_dist
plot = torch.stack(out).flatten(1).T
fig = plot_dist(plot.detach().numpy(), times=(-0.4, 1.4))
fig.title.set_text("Decoding accuracy")
