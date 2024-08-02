import torch
from analysis.grouping import GroupData
from ieeg.viz.ensemble import plot_dist, subgrids, plot_weight_dist
import os
import numpy as np
import matplotlib.pyplot as plt
import slicetca
from multiprocessing import freeze_support
from scipy import stats as st
from ieeg.calc.mat import LabeledArray
from sklearn.preprocessing import minmax_scale
from functools import partial
from copy import deepcopy
from tslearn import metrics
from slicetca.core.helper_functions import squared_difference, to_sparse


# ## set up the figure
fpath = os.path.expanduser("~/Box/CoganLab")
# # Create a gridspec instance with 3 rows and 3 columns
device = ('cuda' if torch.cuda.is_available() else 'cpu')


def mse(x, y, w=None):
    out = (x - y) ** 2
    if w is not None:
        out2 = out * w
        out = out2 * out.sum() / out2.sum()
    return out
# %% Load the data

if __name__ == '__main__':
    freeze_support()
    sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats')
    # %%
    idx = sorted(list(sub.SM))
    zscore = np.nanmean(sub['zscore'].array, axis=(-4, -2))
    aud_slice = slice(0, 175)
    pval = sub.p_vals
    # pval = np.hstack([pval['aud_ls', :, aud_slice],
                     # pval['resp']])
    pval = np.where(pval > 0.9999, 0.9999, pval,)
    zpval = LabeledArray(st.norm.ppf(1 - pval), zscore.labels)
    pval = LabeledArray(pval, zscore.labels)
    met = zscore
    trainz = np.hstack([met['aud_ls', :, aud_slice],
                        met['aud_lm',:, aud_slice],
                        met['go_ls'],
                        met['resp']])
    stitched = np.hstack([sub.signif['aud_ls', :, aud_slice],
                          sub.signif['aud_lm', :, aud_slice],
                          sub.signif['go_ls'],
                          sub.signif['resp', :]])
    weights = torch.as_tensor(1 - np.hstack([pval['aud_ls', :, aud_slice],
                            pval['aud_lm', :, aud_slice],
                            pval['go_ls'],
                            pval['resp']]))
    data = trainz[idx]
    neural_data_tensor = torch.from_numpy(data
                                          / np.nanstd(data)
                                          ).to(device)

    # Assuming neural_data_tensor is your 3D tensor
    # Remove NaN values
    mask = torch.from_numpy(stitched[idx]).to(device)
    # masked_data = torch.masked.masked_tensor(neural_data_tensor, mask.type(torch.bool), requires_grad=True)
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
    # test_mask = torch.logical_and(test_mask, mask)
    # train_mask = torch.logical_and(train_mask, mask)

    procs = 3
    torch.set_num_threads(2)
    min_ranks = [1]
    reps = 40
    loss_grid, seed_grid = slicetca.grid_search(neural_data_tensor,
                                                min_ranks=min_ranks,
                                                max_ranks=[10],
                                                sample_size=reps,
                                                mask_train=train_mask,
                                                mask_test=test_mask,
                                                processes_grid=procs,
                                                seed=1,
                                                min_std=10 ** -4,
                                                learning_rate=5 * 10 ** -3,
                                                max_iter=10**4,
                                                positive=True,
                                                batch_prop=0.2,
                                                batch_prop_decay=3,
                                                initialization='uniform-positive',
                                                init_bias=0.001,
                                                loss_function=partial(mse, w=weights[idx]))
    #     seed_grid = data['seed_grid']
    x_ticks = np.arange(0, 10)
    x_data = np.repeat(x_ticks, reps)
    y_data = loss_grid.flatten()
    ax = plot_dist(np.squeeze(loss_grid).T)
    ax.scatter(x_data, y_data, c='k')
    plt.xticks(x_ticks, np.arange(1, 11))
    #
    # # %% decompose the optimal model
    # n_components = (np.unravel_index(loss_grid.argmin(), loss_grid.shape) + np.array([1, 0]))[:-1]
    n_components = np.mean(loss_grid, axis=1).argmin() + 1
    # best_seed = seed_grid[np.unravel_index(loss_grid.argmin(), loss_grid.shape)]
    best_seed = seed_grid[loss_grid[:, n_components - 1].argmin(), n_components-1]
    # with torch.profiler.profile(
    #         schedule=torch.profiler.schedule(wait=0, warmup=1, active=1,
    #                                          repeat=0),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    #         record_shapes=True,
    #         with_stack=True,
    #         activities=[torch.profiler.ProfilerActivity.CPU]
    # ) as prof:
    #     for i in range(2):
    # n_components = 4
    # os.environ['TORCH_LOGS'] = 'not_implemented'
    # import logging
    #
    # torch._logging.set_logs(all=logging.DEBUG)
    # with torch.autograd.detect_anomaly(True):
    losses, model = slicetca.decompose(
        neural_data_tensor,
        # masked_data,
        #                                [3],
                                       [n_components],
                               seed=best_seed,
                               positive=True,
                               min_std=10 ** -5,
                               learning_rate=5 * 10 ** -4,
                               max_iter=10 ** 5,
                               # mask=mask.type(torch.bool),
                               batch_prop=0.3,
                               batch_prop_decay=10,
                               initialization='uniform-positive',
        init_bias=0.001,
        loss_function=partial(mse, w=weights[idx]))
    orig = deepcopy(model)
    slicetca.invariance(orig, L2='L2')
            # prof.step()
    W, H = model.get_components(numpy=True)[0]
    # W, H = orig.get_components(numpy=True)[0]

    # %% plot the losses
    plt.figure(figsize=(4, 3), dpi=100)
    plt.plot(np.arange(5000, len(model.losses)), model.losses[5000:], 'k')
    plt.xlabel('iterations')
    plt.ylabel('mean squared error')
    plt.xlim(0, len(model.losses))
    plt.tight_layout()
    # %% plot the model
    axes = slicetca.plot(model,
                         variables=('trial', 'time', 'neuron'),)

    # %%
    met = zscore
    plotz = np.hstack([met['aud_ls', :, aud_slice],
                        met['resp']])
    # plotz /= (std := np.nanstd(plotz))
    colors = ['b', 'r', 'g', 'y', 'k']
    fig = plot_weight_dist(plotz[idx], W.T, colors=colors)

    # %%
    # fig, axs = plt.subplots(2, 2)
    size = minmax_scale(W, feature_range=(0.05, 0.3))
    for i in range(W.shape[0]):
        # ax = axs.flatten()[i]
        size = W[i]
        sub.plot_groups_on_average([idx], size=list(size), hemi='lh', colors=[colors[i]])