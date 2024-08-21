# %% Imports
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import torch
import numpy as np
from analysis.decoding.models import CNNTransformer, SimpleDecoder
from analysis.grouping import GroupData
import os
from ieeg.calc.mat import Labels, LabeledArray

# %% Define data module
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
    data / np.nanstd(data)).to(device).permute(1, 2, 0)

# %% Create model
# model parameters
in_channels = data.shape[0]
num_classes = 4
d_model = data.shape[1]
kernel_time = 50  # ms
kernel_size = int(kernel_time * fs / 1000)  # kernel length in samples
stride_time = 10  # ms
stride = int(stride_time * fs / 1000)  # stride length in samples
padding = 0
n_head = 6
num_layers = 3
dim_fc = 128
dropout = 0.3
learning_rate = 1e-3

# instantiate the model
model = SimpleDecoder(in_channels, num_classes, d_model, kernel_size, stride, padding,
                      learning_rate)

# %% Train the model with kfold
# instantiate the trainer
n_folds = 5
max_epochs = 10

callbacks = [ModelCheckpoint(monitor='val_loss'), EarlyStopping(monitor='val_loss', patience=3)]
trainer = L.Trainer(max_epochs=max_epochs,
                    accelerator='cpu',
                    # callbacks=callbacks,
                    logger=True,
                    )
target_map = {'heat': 0, 'hut': 1, 'hot': 2, 'hoot': 3}
target = torch.as_tensor([target_map[l[1]] for l in data.labels[1].split('-')])
trainer.fit(model, (neural_data_tensor[None].type(torch.float32), target[None]))
# # train the model
# for fold in range(n_folds):
#     data_module.set_fold(fold)
#     model.current_fold = fold
#     trainer.fit(model, data_module)
#     trainer.test(model, data_module.test_dataloader())