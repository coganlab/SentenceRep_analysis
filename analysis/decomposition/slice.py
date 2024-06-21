import torch
from analysis.grouping import GroupData
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
    idx = sorted(list(sub.sig_chans))
    aud_slice = slice(0, 175)
    reduced = sub[:, :, :, idx][:, ('aud_ls', 'aud_lm', 'aud_jl', 'go_ls', 'go_lm', 'go_jl'),]
    reduced.array = reduced.array.dropna()
    reduced = reduced.nan_common_denom(True, 10, True)
    idx = [i for i, l in enumerate(sub.array.labels[3]) if
     l in reduced.array.labels[2]]
    # transfer data to torch tensor
    conds_aud = ['aud_ls', 'aud_lm', 'aud_jl']
    conds_go = ['go_ls', 'go_lm', 'go_jl']
    aud = reduced.array['zscore', conds_aud]
    aud.labels[0] = aud.labels[0].replace("aud_", "")
    go = reduced.array['zscore', conds_go]
    go.labels[0] = go.labels[0].replace("go_", "")
    aud_go = aud[..., :175].concatenate(go, -1)
    data = aud_go.combine((0,1)).combine((0, 2)).__array__()
    # del sub

    stitched = np.hstack([sub.signif['aud_ls', :, aud_slice],
                          # sub.signif['aud_lm', :, aud_slice],
                          sub.signif['resp', :]])
    neural_data_tensor = torch.from_numpy(
        (data / np.std(data, axis=0)).swapaxes(0, 1)).to(device)

    # Assuming neural_data_tensor is your 3D tensor
    # Remove NaN values
    mask = ~torch.isnan(neural_data_tensor)
    # neural_data_tensor[mask.any(dim=2)] = 0.
    #
    # # Convert to sparse tensor
    # sparse_tensor = neural_data_tensor.to_sparse(sparse_dim=2)
    # del data

    ## set up the model

    train_mask, test_mask = slicetca.block_mask(dimensions=neural_data_tensor.shape,
                                                train_blocks_dimensions=(1, 1, 10), # Note that the blocks will be of size 2*train_blocks_dimensions + 1
                                                test_blocks_dimensions=(1, 1, 5), # Same, 2*test_blocks_dimensions + 1
                                                fraction_test=0.2,
                                                device=device)

    procs = 3
    torch.set_num_threads(joblib.cpu_count() // procs)
    # loss_grid, seed_grid = slicetca.grid_search(neural_data_tensor,
    #                                             min_ranks = [0, 0, 0],
    #                                             max_ranks = [4, 10, 7],
    #                                             sample_size=4,
    #                                             mask_train=train_mask,
    #                                             mask_test=test_mask,
    #                                             processes_grid=procs,
    #                                             seed=1,
    #                                             min_std=10**-4,
    #                                             learning_rate=5*10**-3,
    #                                             max_iter=10**4,
    #                                             positive=True)
    np.savez('../loss_grid.npz', loss_grid=loss_grid, seed_grid=seed_grid,
             idx=idx)
    slicetca.plot_grid(loss_grid, min_ranks=(0, 0, 1))
    # load the grid
    with np.load('../loss_grid.npz') as data:
        loss_grid = data['loss_grid']
        seed_grid = data['seed_grid']

    # %% decompose the optimal model
    n_components = (np.unravel_index(loss_grid.argmin(), loss_grid.shape) + np.array([0, 0, 1, 0]))[:-1]
    best_seed = seed_grid[np.unravel_index(loss_grid.argmin(), loss_grid.shape)]
    losses, model = slicetca.decompose(neural_data_tensor, n_components,
                               seed=best_seed,
                               positive=True,
                               min_std=10 ** -4,
                               learning_rate=5 * 10 ** -3,
                               max_iter=10 ** 4
                               )


    # %% plot the losses
    plt.figure(figsize=(4, 3), dpi=100)
    plt.plot(np.arange(500, len(model.losses)), model.losses[500:], 'k')
    plt.xlabel('iterations')
    plt.ylabel('mean squared error')
    plt.xlim(0, len(model.losses))
    plt.tight_layout()
    # %% plot the model
    axes = slicetca.plot(model,
                         variables=('trial', 'neuron', 'time'),)
