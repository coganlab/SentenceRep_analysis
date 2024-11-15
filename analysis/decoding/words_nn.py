import torch
import numpy as np
import os
from analysis.decoding.models import SimpleDecoder
from analysis.decoding.train import process_data
from ieeg.calc.mat import LabeledArray
from analysis.decoding import windower
from joblib import Parallel, delayed
import logging
import matplotlib.pyplot as plt
from analysis.grouping import GroupData
from analysis.data import dataloader


model = torch.load('model1.pt')
n_components = 5

# %% windowed decoding aud

# load the data
fpath = os.path.expanduser("~/Box/CoganLab")
sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats')
sm_idx = sorted(list(sub.SM))
aud_slice = slice(0, 175)
conds = ['aud_ls', 'aud_lm', 'go_ls', 'go_lm', 'resp']
neural_data_tensor, labels = dataloader(sub, sm_idx, conds)
mask = ~torch.isnan(neural_data_tensor)
neural_data_tensor, _ = dataloader(sub, sm_idx, conds, do_mixup=True)

n_folds = 5
val_size = 1 / n_folds
max_epochs = 500
results = {str(i): None for i in range(n_components)}
target_map = {'heat': 0, 'hut': 1, 'hot': 2, 'hoot': 3}
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
decoders = [SimpleDecoder(4, len(labels[2]), 5e-4) for _ in range(n_components[0])]
for i, decoder in enumerate(decoders):
    # xi = model.construct_single_component(0, i).detach().cpu().numpy().swapaxes(0, 1)
    xi = neural_data_tensor.clone().detach().cpu().numpy().swapaxes(0, 1)
    trimmed = xi[(W[i] / W.sum(0)) > 0.4]
    sorted_trimmed = trimmed[np.argsort(W[i, (W[i] / W.sum(0)) > 0.4])][
                     ::-1]
    ls = sorted_trimmed[..., :200]
    lm = sorted_trimmed[..., 200:400]
    stacked = np.concatenate([ls, lm], axis=1)
    data_windowed = LabeledArray(windower(stacked, 20, 2).swapaxes(0, -1))[::5]
    data_windowed.labels[2] = np.concatenate([labels[0], labels[0]])
    data_windowed.labels[1] = labels[1]
    # decoder = SimpleDecoder(4, xi.shape[1] * xi.shape[2], 5e-3)
    # train the decoder briefly
    out = Parallel(n_jobs=-2, verbose=40)(delayed(process_data)(
        d, 5, n_folds, val_size, target_map, max_epochs) for d in
                                          data_windowed)
    results[str(i)] = [o.tolist() for o in out]

# %% plot the results
fig, ax = plt.subplots(1, 1)
for i, res in results.items():
    plot = torch.tensor(res).flatten(1).T
    fig = plot_dist(plot.detach().cpu().numpy(), times=(-0.4, 1.4), ax=ax)
    fig.title.set_text("Decoding accuracy")