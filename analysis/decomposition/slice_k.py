import torch
from analysis.grouping import GroupData
from analysis.data import dataloader
import os
import numpy as np
import matplotlib.pyplot as plt
import slicetca
from multiprocessing import freeze_support
from ieeg.viz.ensemble import plot_dist
from slicetca.invariance.iterative_invariance import within_invariance
from lightning.pytorch import Trainer
from analysis.decoding.models import SimpleDecoder
from analysis.decoding.train import process_data
from ieeg.calc.mat import LabeledArray
from analysis.decoding import windower
from joblib import Parallel, delayed
import logging

import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['C:\\Users\\Jakda\\git'])

# ## set up the figure
fpath = os.path.expanduser("~/Box/CoganLab")
# # Create a gridspec instance with 3 rows and 3 columns
#

# %% Load the data

if __name__ == '__main__':
    freeze_support()
    sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats')
    idx = sorted(list(sub.SM))
    aud_slice = slice(0, 175)
    conds = ['aud_ls', 'aud_lm', 'go_ls', 'go_lm', 'resp']
    neural_data_tensor, labels = dataloader(sub, idx, conds)
    mask = ~torch.isnan(neural_data_tensor)
    neural_data_tensor[torch.isnan(neural_data_tensor)] = 0

    ## set up the model

    train_mask, test_mask = slicetca.block_mask(dimensions=neural_data_tensor.shape,
                                                train_blocks_dimensions=(1, 1, 10), # Note that the blocks will be of size 2*train_blocks_dimensions + 1
                                                test_blocks_dimensions=(1, 1, 5), # Same, 2*test_blocks_dimensions + 1
                                                fraction_test=0.2)
    test_mask = torch.logical_and(test_mask, mask)
    train_mask = torch.logical_and(train_mask, mask)

    procs = 2
    # torch.set_num_threads(6)
    threads = 2
    min_ranks = [1]
    max_ranks = [8]
    repeats = 2
    loss_grid, seed_grid = slicetca.grid_search(neural_data_tensor,
                                                min_ranks = min_ranks,
                                                max_ranks = max_ranks,
                                                sample_size=repeats,
                                                mask_train=mask,
                                                mask_test=None,
                                                processes_grid=procs,
                                                processes_sample=threads,
                                                seed=1,
                                                # batch_prop=0.33,
                                                # batch_prop_decay=3,
                                                # min_std=1e-4,
                                                # iter_std=10,
                                                init_bias=0.01,
                                                # weight_decay=1e-3,
                                                initialization='orthogonal',
                                                learning_rate=1e-2,
                                                max_iter=10000,
                                                positive=True,
                                                verbose=0,
                                                # batch_dim=0,
                                                loss_function=torch.nn.HuberLoss(reduction='sum'),)
    # np.savez('../loss_grid.npz', loss_grid=loss_grid, seed_grid=seed_grid,
    #          idx=idx)
    # slicetca.plot_grid(loss_grid, min_ranks=(0, 1, 0))
    # load the grid
    # with np.load('../loss_grid.npz') as data:
    #     loss_grid = data['loss_grid']
    #     seed_grid = data['seed_grid']
    x_data = np.repeat(np.arange(max_ranks[0]), repeats)
    y_data = loss_grid.flatten()
    ax = plot_dist(np.atleast_2d(np.squeeze(loss_grid).T))
    ax.scatter(x_data, y_data, c='k')
    plt.xticks(np.arange(max_ranks[0]), np.arange(max_ranks[0]) + min_ranks[0])

    import pickle
    with open('results_grid.pkl', 'wb') as f:
        pickle.dump({'loss': loss_grid.tolist(), 'seed': seed_grid.tolist()}, f)