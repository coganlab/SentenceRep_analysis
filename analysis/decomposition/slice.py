import torch
from analysis.grouping import GroupData
import os
import numpy as np
import matplotlib.pyplot as plt
import slicetca
from multiprocessing import freeze_support
from ieeg.viz.ensemble import plot_dist
from slicetca.core.helper_functions import huber_loss, to_sparse
from functools import partial
import pyvistaqt as pv

# ## set up the figure
fpath = os.path.expanduser("~/Box/CoganLab")
# # Create a gridspec instance with 3 rows and 3 columns
device = ('cuda' if torch.cuda.is_available() else 'cpu')
#


# %% Load the data

if __name__ == '__main__':
    freeze_support()
    sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats')
    idx = sorted(list(sub.SM))
    aud_slice = slice(0, 175)
    reduced = sub[:, :, :, idx][:, ('aud_ls', 'go_ls'),]
    reduced.array = reduced.array.dropna()
    # reduced = reduced.nan_common_denom(True, 10, True)
    # idx = [i for i, l in enumerate(sub.array.labels[3]) if
    #  l in reduced.array.labels[2]]
    # transfer data to torch tensor
    conds_aud = ['aud_ls']
    conds_go = ['go_ls']
    aud = reduced.array['zscore', conds_aud]
    aud.labels[0] = aud.labels[0].replace("aud_", "")
    go = reduced.array['zscore', conds_go]
    go.labels[0] = go.labels[0].replace("go_", "")
    aud_go = aud[..., :175].concatenate(go, -1)
    # aud_go = LabeledArray(np.ascontiguousarray(aud_go.__array__()), aud_go.labels)
    data = aud_go.combine((0, 1)).combine((0, 2)).__array__()
    # del sub

    stitched = np.hstack([sub.signif['aud_ls', :, aud_slice],
                          # sub.signif['aud_lm', :, aud_slice],
                          sub.signif['resp', :]])

    neural_data_tensor = torch.from_numpy(
        (data / np.nanstd(data)).swapaxes(0, 1)).to(device)

    # Assuming neural_data_tensor is your 3D tensor
    # Remove NaN values
    mask = ~torch.isnan(neural_data_tensor)
    # mask = mask & torch.from_numpy(stitched[None, idx]).to(device, dtype=torch.bool)
    # offset = torch.min(neural_data_tensor[mask]) - 1
    # neural_data_tensor -= offset
    # neural_data_tensor[~mask] = 0.
    #
    # # Convert to sparse tensor
    # sparse_tensor = to_sparse(neural_data_tensor, mask)
    neural_data_tensor[mask == 0] = 0
    # del data

    ## set up the model
    grid = False
    if grid:
        train_mask, test_mask = slicetca.block_mask(dimensions=neural_data_tensor.shape,
                                                    train_blocks_dimensions=(1, 1, 10), # Note that the blocks will be of size 2*train_blocks_dimensions + 1
                                                    test_blocks_dimensions=(1, 1, 5), # Same, 2*test_blocks_dimensions + 1
                                                    fraction_test=0.2,
                                                    device=device)
        test_mask = torch.logical_and(test_mask, mask)
        train_mask = torch.logical_and(train_mask, mask)

        procs = 2
        torch.set_num_threads(2)
        threads = 2
        min_ranks = [0, 1, 0]
        loss_grid, seed_grid = slicetca.grid_search(neural_data_tensor,
                                                    min_ranks = min_ranks,
                                                    max_ranks = [0,8,0],
                                                    sample_size=30,
                                                    mask_train=train_mask,
                                                    mask_test=test_mask,
                                                    processes_grid=procs,
                                                    processes_sample=threads,
                                                    seed=1,
                                                    min_std=10 ** -4,
                                                    learning_rate=5*10 ** -3,
                                                    max_iter=10 ** 4,
                                                    positive=True,
                                                    batch_prop=0.2,
                                                    batch_prop_decay=3,
                                                    loss_function=partial(huber_loss, delta=1.0),)
        # np.savez('../loss_grid.npz', loss_grid=loss_grid, seed_grid=seed_grid,
        #          idx=idx)
        slicetca.plot_grid(loss_grid, min_ranks=(0, 1, 0))
        # load the grid
        # with np.load('../loss_grid.npz') as data:
        #     loss_grid = data['loss_grid']
        #     seed_grid = data['seed_grid']
        x_ticks = np.arange(0, 10)
        x_data = np.repeat(x_ticks, 10)
        y_data = loss_grid.flatten()
        ax = plot_dist(np.squeeze(loss_grid).T)
        ax.scatter(x_data, y_data, c='k')
        plt.xticks(x_ticks, np.arange(1, 11))

        n_components = (np.unravel_index(loss_grid.argmin(), loss_grid.shape) + np.array([0, 1, 0, 0]))[:-1]
        best_seed = seed_grid[np.unravel_index(loss_grid.argmin(), loss_grid.shape)]
    else:
        n_components = (5, 0)
        best_seed = 123456

    # #
    # %% decompose the optimal model
    # n_components = (np.unravel_index(loss_grid.argmin(), loss_grid.shape) + np.array([0, 1, 0, 0]))[:-1]
    # best_seed = seed_grid[np.unravel_index(loss_grid.argmin(), loss_grid.shape)]
    # with torch.autograd.profiler.profile(with_modules=True) as prof:
    def loss(x, y):
        # p = torch.nn.LogSoftmax(dim=1)(x)
        # q = torch.nn.Softmax(dim=1)(y)
        # return torch.nn.KLDivLoss(reduction='none')(p, q)
        return (x - y) ** 2

    losses, model = slicetca.decompose(neural_data_tensor,
                                       n_components,
                                       seed=best_seed,
                                       positive=True,
                                       min_std=10 ** -6,
                                       iter_std=100,
                                       learning_rate=5 * 10 ** -3,
                                       max_iter=10 ** 4,
                                       batch_dim=0,
                                       # batch_prop_decay=3,
                                       mask=mask,
                                       initialization='uniform-positive',
                                       loss_function=huber_loss,)
                                       # verbose=True,)
    # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    # slicetca.invariance(model, L3 = None)
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
    colors = ['b', 'r', 'g', 'y', 'k', 'c', 'm']
    W, H = model.get_components(numpy=True)[0]
    fig, ax = plt.subplots(1, 1)
    for i in range(W.shape[0]):
        plot_dist(model.construct_single_component(0, i).detach().numpy()[(W[i] / W.sum(0)) > 0.1],
                  ax=ax, color=colors[i])

    # %%
    min_size = int(np.ceil(np.sqrt(W.shape[0])))
    min_size = [int(np.ceil(np.sqrt(W.shape[0] / min_size))), min_size]
    plotter = pv.BackgroundPlotter(shape=min_size)
    size = W.copy()
    size[size > 2] = 2
    for i in range(W.shape[0]):
        j, k = divmod(i, min_size[1])
        plotter.subplot(j, k)
        brain = sub.plot_groups_on_average([idx], size=list(size[i]), hemi='both',
                                           colors=[colors[i]], show=False)
        for actor in brain.plotter.actors.values():
            plotter.add_actor(actor, reset_camera=False)
        plotter.camera = brain.plotter.camera
        plotter.camera_position = brain.plotter.camera_position
    plotter.link_views()