import torch
from analysis.grouping import GroupData
from analysis.data import dataloader
import os
import numpy as np
import matplotlib.pyplot as plt
import slicetca
from multiprocessing import freeze_support
from ieeg.viz.ensemble import plot_dist


# ## set up the figure
fpath = os.path.expanduser("~/Box/CoganLab")
# # Create a gridspec instance with 3 rows and 3 columns
#

# %% Load the data

if __name__ == '__main__':
    freeze_support()
    sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats')
    groups = ['AUD', 'SM', 'PROD']
    torch.set_float32_matmul_precision('medium')
    for group in groups:
        idx = sorted(list(set(getattr(sub, group))))
        aud_slice = slice(0, 175)
        conds = ['aud_ls', 'aud_lm', 'go_ls', 'go_lm', 'resp']
        neural_data_tensor, labels = dataloader(sub, idx, conds)
        mask = ~torch.isnan(neural_data_tensor)
        neural_data_tensor, _ = dataloader(sub, idx, conds, do_mixup=True)
        neural_data_tensor = neural_data_tensor.to(torch.float32)

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
        min_ranks = [0, 1, 0]
        max_ranks = [0, 8, 0]
        repeats = 4
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        loss_grid, seed_grid = slicetca.grid_search(neural_data_tensor,
                                                    min_ranks = min_ranks,
                                                    max_ranks = max_ranks,
                                                    sample_size=repeats,
                                                    mask_train=mask,
                                                    mask_test=None,
                                                    processes_grid=procs,
                                                    processes_sample=threads,
                                                    seed=3,
                                                    batch_prop=0.2,
                                                    # batch_prop_decay=3,
                                                    # min_std=1e-4,
                                                    # iter_std=10,
                                                    init_bias=0.01,
                                                    # weight_decay=1e-3,
                                                    initialization='uniform-positive',
                                                    learning_rate=1e-2,
                                                    max_iter=10000,
                                                    positive=True,
                                                    verbose=0,
                                                    # batch_dim=0,
                                                    loss_function=torch.nn.HuberLoss(reduction='mean'),)
                                                    # compile=True)
        # np.savez('../loss_grid.npz', loss_grid=loss_grid, seed_grid=seed_grid,
        #          idx=idx)
        # slicetca.plot_grid(loss_grid, min_ranks=(0, 1, 0))
        # load the grid
        # with np.load('../loss_grid.npz') as data:
        #     loss_grid = data['loss_grid']
        #     seed_grid = data['seed_grid']
        try:
            max_r = max(max_ranks)
            min_r = min(min_ranks)
            x_data = np.repeat(np.arange(max_r), repeats)
            y_data = loss_grid.flatten()
            ax = plot_dist(np.atleast_2d(np.squeeze(loss_grid).T))
            ax.scatter(x_data, y_data, c='k')
            plt.xticks(np.arange(max_r), np.arange(max_r) + min_r)
            plt.title(f"Loss for {group}")
            plt.savefig(f"loss_dist_{group}.png")
        except:
            pass

        import pickle
        with open(f"results_grid_{group}.pkl", 'wb') as f:
            pickle.dump({'loss': loss_grid.tolist(), 'seed': seed_grid.tolist()}, f)

        n_components = (np.unravel_index(np.argmin(loss_grid), loss_grid.shape))[0] + 1
        best_seed = seed_grid[n_components - 1, np.argmin(loss_grid[n_components - 1])]