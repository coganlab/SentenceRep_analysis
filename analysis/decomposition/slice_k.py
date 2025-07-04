import torch
from analysis.grouping import GroupData
from analysis.data import dataloader
import os
import numpy as np
import matplotlib.pyplot as plt
import slicetca
from multiprocessing import freeze_support
from ieeg.viz.ensemble import plot_dist
from itertools import product
from functools import partial


# ## set up the task
HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    n = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    n = 1

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

# %% Load the data

if __name__ == '__main__':
    freeze_support()
    sub = GroupData.from_intermediates("SentenceRep", LAB_root, folder='stats')
    param_grid = {'ranks': [{'min': [1, 0], 'max': [8, 0]},
                            {'min': [1], 'max': [8]},],
                  'groups': ['AUD', 'SM', 'PROD', 'sig_chans'],
                  'masks': [{'train': False, 'test': False},],
                            # {'train': True, 'test': True},]
                  'loss': ['HuberLoss'],
                  'lr': [1e-3],
                  'decay': [.1]
                    }
    procs = 1
    threads = 1
    repeats = 5
    conds = ['aud_ls', 'aud_lm', 'go_ls', 'go_lm']
    aud_slice = slice(0, 175)

    for ranks, group, is_mask, loss, lr, decay in product(
            param_grid['ranks'], param_grid['groups'], param_grid['masks'],
            param_grid['loss'], param_grid['lr'], param_grid['decay']):
        if n > 1:
            n -= 1
            continue
        idx = sorted(list(set(getattr(sub, group))))
        neural_data_tensor, labels = dataloader(sub.array, idx, conds)
        mask = ~torch.isnan(neural_data_tensor)
        # neural_data_tensor, _ = dataloader(sub, idx, conds, do_mixup=False)
        neural_data_tensor = neural_data_tensor.to(torch.float32)

        ## set up the model
        if is_mask['train'] and is_mask['test']:
            train_mask, test_mask = slicetca.block_mask(dimensions=neural_data_tensor.shape,
                                                        train_blocks_dimensions=(1, 1, 10), # Note that the blocks will be of size 2*train_blocks_dimensions + 1
                                                        test_blocks_dimensions=(1, 1, 5), # Same, 2*test_blocks_dimensions + 1
                                                        fraction_test=0.2)
            test_mask = torch.logical_and(test_mask, mask)
            train_mask = torch.logical_and(train_mask, mask)
        else:
            train_mask = mask
            test_mask = None

        loss_grid, seed_grid = slicetca.grid_search(neural_data_tensor,
                                                    min_ranks = ranks['min'],
                                                    max_ranks = ranks['max'],
                                                    sample_size=repeats,
                                                    mask_train=train_mask,
                                                    mask_test=test_mask,
                                                    processes_grid=procs,
                                                    processes_sample=threads,
                                                    seed=3,
                                                    batch_prop=decay,
                                                    batch_prop_decay=5 if decay < 1 else 1,
                                                    # min_std=1e-4,
                                                    # iter_std=10,
                                                    init_bias=0.01,
                                                    weight_decay=partial(torch.optim.Adam,
                                betas=(0.5, 0.5),
                                amsgrad=True,
                                eps=1e-10,
                                weight_decay=0
                             ),
                                                    initialization='uniform-positive',
                                                    learning_rate=lr,
                                                    max_iter=1000000,
                                                    positive=True,
                                                    verbose=0,
                                                    batch_dim=0,
                                                    shuffle_dim=1,
                                                    loss_function=getattr(torch.nn, loss)(reduction='mean'),
                                                    compile=True)

        max_r = max(ranks['max'])
        min_r = min(ranks['min'])
        x_data = np.repeat(np.arange(max_r), repeats)
        y_data = loss_grid.flatten()
        ax = plot_dist(np.atleast_2d(np.squeeze(loss_grid).T))
        ax.scatter(x_data, y_data, c='k')
        plt.xticks(np.arange(max_r), np.arange(max_r) + min_r)
        plt.title(f"Loss for {group}")

        import pickle
        file_id = group
        file_id += f"_{len(ranks['min'])}ranks"
        file_id += "_test" if is_mask['test'] else ""
        file_id += f"_{loss}"
        file_id += f"_{lr}"
        file_id += f"_{decay}"
        plt.savefig(f"loss_dist_{file_id}.png")
        with open(f"results_grid_{file_id}.pkl", 'wb') as f:
            pickle.dump({'loss': loss_grid.tolist(), 'seed': seed_grid.tolist()}, f)

        if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
            break

        # n_components = (np.unravel_index(np.argmin(loss_grid), loss_grid.shape))[0] + 1
        # best_seed = seed_grid[n_components - 1, np.argmin(loss_grid[n_components - 1])]