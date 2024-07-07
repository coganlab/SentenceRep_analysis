import torch
from analysis.grouping import GroupData
from ieeg.viz.ensemble import plot_dist, subgrids
import os
import numpy as np
import matplotlib.pyplot as plt
import slicetca
from multiprocessing import freeze_support
import joblib
from scipy.sparse import csr_matrix

# ## set up the figure
fpath = os.path.expanduser("~/Box/CoganLab")
# # Create a gridspec instance with 3 rows and 3 columns
device = ('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# %% Load the data

if __name__ == '__main__':
    freeze_support()
    sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats')
    idx = sorted(list(sub.SM))
    zscore = np.nanmean(sub['zscore'].array, axis=(-4, -2))
    aud_slice = slice(0, 175)
    met = zscore
    trainz = np.hstack([met['aud_ls', :, aud_slice],
                        met['resp']])
    stitched = np.hstack([sub.signif['aud_ls', :, aud_slice],
                          # sub.signif['aud_lm', :, aud_slice],
                          sub.signif['resp', :]])
    neural_data_tensor = torch.from_numpy(trainz[idx]).to(device)

    # Assuming neural_data_tensor is your 3D tensor
    # Remove NaN values
    mask = torch.from_numpy(stitched[idx]).to(device)
    # neural_data_tensor[mask.any(dim=2)] = 0.
    #
    # # Convert to sparse tensor
    # sparse_tensor = neural_data_tensor.to_sparse(sparse_dim=2)
    # del data

    ## set up the model

    train_mask, test_mask = slicetca.block_mask(dimensions=neural_data_tensor.shape,
                                                train_blocks_dimensions=(1, 10), # Note that the blocks will be of size 2*train_blocks_dimensions + 1
                                                test_blocks_dimensions=(1, 5), # Same, 2*test_blocks_dimensions + 1
                                                fraction_test=0.2,
                                                device=device)
    test_mask = torch.logical_and(test_mask, mask)
    train_mask = torch.logical_and(train_mask, mask)

    procs = 1
    torch.set_num_threads(1)
    min_ranks = [1]
    loss_grid, seed_grid = slicetca.grid_search(neural_data_tensor,
                                                min_ranks = min_ranks,
                                                max_ranks = [10],
                                                sample_size=4,
                                                mask_train=train_mask,
                                                mask_test=test_mask,
                                                processes_grid=procs,
                                                seed=2,
                                                min_std=10 ** -4,
                                                learning_rate=5 * 10 ** -3,
                                                max_iter=10**4,
                                                positive=True)
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # np.savez('../loss_grid.npz', loss_grid=loss_grid, seed_grid=seed_grid,
    #          idx=idx)
    # slicetca.plot_grid(np.mean(loss_grid, axis=0), min_ranks=(0,))
    # load the grid
    # with np.load('../loss_grid.npz') as data:
    #     loss_grid = data['loss_grid']
    #     seed_grid = data['seed_grid']
    # from ieeg.viz.ensemble import plot_dist;
    #
    plot_dist(loss_grid.T[:, 1:])

    # %% decompose the optimal model
    n_components = (np.unravel_index(loss_grid.argmin(), loss_grid.shape) + np.array([0, 0]))[:-1]
    best_seed = seed_grid[np.unravel_index(loss_grid.argmin(), loss_grid.shape)]
    # with torch.profiler.profile(
    #         schedule=torch.profiler.schedule(wait=0, warmup=1, active=1,
    #                                          repeat=0),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    #         record_shapes=True,
    #         with_stack=True,
    #         activities=[torch.profiler.ProfilerActivity.CPU]
    # ) as prof:
    #     for i in range(2):
    n_components = 6
    losses, model = slicetca.decompose(neural_data_tensor, n_components,
                               seed=best_seed,
                               positive=True,
                               min_std=10 ** -4,
                               learning_rate=5 * 10 ** -3,
                               max_iter=10 ** 5
                               )
    slicetca.invariance(model)
            # prof.step()

    # %% plot the losses
    plt.figure(figsize=(4, 3), dpi=100)
    plt.plot(np.arange(500, len(model.losses)), model.losses[500:], 'k')
    plt.xlabel('iterations')
    plt.ylabel('mean squared error')
    plt.xlim(0, len(model.losses))
    plt.tight_layout()
    # %% plot the model
    axes = slicetca.plot(model,
                         variables=('neuron', 'time'),)
