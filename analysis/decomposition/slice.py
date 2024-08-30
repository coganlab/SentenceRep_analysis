import torch
from analysis.grouping import GroupData
import os
import numpy as np
import matplotlib.pyplot as plt
import slicetca
from multiprocessing import freeze_support
from ieeg.viz.ensemble import plot_dist
from functools import partial, reduce
from ieeg.calc.fast import mixup
from slicetca.invariance.iterative_invariance import within_invariance
from lightning.pytorch import Trainer, utilities, loggers
from analysis.decoding.models import SimpleDecoder
import logging

import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['C:\\Users\\Jakda\\git'])

# ## set up the figure
fpath = os.path.expanduser("~/Box/CoganLab")
# # Create a gridspec instance with 3 rows and 3 columns
device = ('cuda' if torch.cuda.is_available() else 'cpu')
#


def dataloader(sub, idx, conds, metric='zscore', do_mixup=False, no_nan=False):
    reduced = sub[:, :, :, idx][:, conds,]
    reduced.array = reduced.array.dropna()
    if no_nan:
        reduced.nan_common_denom(True, 10, True)
    std = np.nanstd(reduced.array[metric].__array__())
    if do_mixup:
        mixup(reduced.array[metric], 3)
    combined = reduce(lambda x, y: x.concatenate(y, -1),
                      [reduced.array[metric, c] for c in conds])
    data = combined.combine((0, 2)).swapaxes(0, 1)
    neural_data_tensor = torch.from_numpy(
        (data.__array__() / std)).to(device)
    return neural_data_tensor, data.labels



# %% Load the data

if __name__ == '__main__':
    freeze_support()
    sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats')
    idx = sorted(list(sub.SM))
    aud_slice = slice(0, 175)
    conds = ['aud_ls', 'aud_lm', 'go_ls', 'resp']
    neural_data_tensor, labels = dataloader(sub, idx, conds)
    mask = ~torch.isnan(neural_data_tensor)
    neural_data_tensor[torch.isnan(neural_data_tensor)] = 0

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

        procs = 1
        # torch.set_num_threads(6)
        threads = 1
        min_ranks = [1]
        repeats = 10
        loss_grid, seed_grid = slicetca.grid_search(neural_data_tensor,
                                                    min_ranks = min_ranks,
                                                    max_ranks = [8],
                                                    sample_size=repeats,
                                                    mask_train=train_mask,
                                                    mask_test=test_mask,
                                                    processes_grid=procs,
                                                    processes_sample=threads,
                                                    seed=1,
                                                    # min_std=10 ** -5,
                                                    # iter_std=100,
                                                    init_bias=0.01,
                                                    weight_decay=0.001,
                                                    initialization='orthogonal',
                                                    learning_rate=5e-2,
                                                    max_iter=10000,
                                                    positive=True,
                                                    verbose=0,
                                                    # batch_dim=0,
                                                    loss_function=torch.nn.MSELoss(),)
        # np.savez('../loss_grid.npz', loss_grid=loss_grid, seed_grid=seed_grid,
        #          idx=idx)
        # slicetca.plot_grid(loss_grid, min_ranks=(0, 1, 0))
        # load the grid
        # with np.load('../loss_grid.npz') as data:
        #     loss_grid = data['loss_grid']
        #     seed_grid = data['seed_grid']
        x_ticks = np.arange(0, 8)
        x_data = np.repeat(x_ticks, repeats-1)
        y_data = loss_grid[:,1:].flatten()
        ax = plot_dist(np.atleast_2d(np.squeeze(loss_grid[:,1:]).T))
        ax.scatter(x_data, y_data, c='k')
        plt.xticks(x_ticks, np.arange(1, x_ticks[-1] + 2))

        n_components = (np.unravel_index(loss_grid.argmin(), loss_grid.shape) + np.array([1, 0]))[:-1]
        best_seed = seed_grid[np.unravel_index(loss_grid.argmin(), loss_grid.shape)]
    else:
        n_components = (5,)
        best_seed = 123457

    # #
    # %% decompose the optimal model

    losses, model = slicetca.decompose(neural_data_tensor,
                                       n_components,
                                       # (0, n_components[0], 0),
                                       seed=best_seed,
                                       positive=True,
                                       min_std=1e-7,
                                       iter_std=10,
                                       learning_rate=5e-3,
                                       max_iter=10000,
                                       # batch_dim=0,
                                       batch_prop=0.2,
                                       batch_prop_decay=3,
                                       mask=mask,
                                       init_bias=0.01,
                                       initialization='orthogonal',
                                       loss_function=torch.nn.HuberLoss(),
                                       verbose=0
                                       )
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
    T, W, H = model.get_components(numpy=True)[0]
    # %% plot the components
    colors = colors[:n_components[0]]
    conds = {'aud_ls': (-0.5, 1.5), 'aud_lm': (-0.5, 1.5),
             'go_ls': (-0.5, 1.5),
             'resp': (-1, 1)}
    fig, axs = plt.subplots(1, 4)

    # make a plot for each condition in conds as a subgrid
    for j, (cond, times) in enumerate(conds.items()):
        ax = axs[j]
        start = 200 * j
        end = start + 200
        for i in range(n_components[0]):
            fig = plot_dist(
                # H[i],
                model.construct_single_component(0, i).detach().cpu().numpy()[:,
                (W[i] / W.sum(0)) > 0.4, start:end].reshape(-1, 200),
                ax=ax, color=colors[i], mode='std', times=times)
        if j == 0:
            ax.legend()
            ax.set_ylabel("Z-Score (V)")
            ylims = ax.get_ylim()
        elif j == 1:
            ax.set_xlabel("Time(s)")
        ax.set_ylim(ylims)
        ax.set_title(cond)

    # %%
    from ieeg.viz.mri import electrode_gradient, plot_on_average
    chans = ['-'.join([f"D{int(ch.split('-')[0][1:])}", ch.split('-')[1]]) for
             ch in sub.array.labels[3][idx]]
    # electrode_gradient(sub.subjects, W, chans, colors, mode='both')

    # %% plot each component
    n = W.shape[0]
    for cond, times in conds.items():
        fig, axs = plt.subplots(2, n)
        timings = {'aud_ls': range(0, 200), 'aud_lm': range(200, 400),
         'go_ls': range(400, 600), 'resp': range(600, 800)}
        data = neural_data_tensor.mean(0).detach().cpu().numpy()
        ylims = [0, 0]
        for i, ax in enumerate(axs[0]):
            component = model.construct_single_component(0, i).detach().numpy()
            trimmed = data[(W[i] / W.sum(0)) > 0.4][:, timings[cond]]
            sorted_trimmed = trimmed[np.argsort(W[i, (W[i] / W.sum(0)) > 0.4])][::-1]
            plot_dist(trimmed.reshape(-1, 200), ax=ax, color=colors[i], mode='std', times=times)
            ylims[1] = max(ax.get_ylim()[1], ylims[1])
            ylims[0] = min(ax.get_ylim()[0], ylims[0])

            axs[1, i].imshow(sorted_trimmed, aspect='auto', cmap='inferno')

        for ax in axs[0]:
            ax.set_ylim(ylims)
        fig.suptitle(cond)

    # %% varimax rotation
    # create a copy of the model
    model_rot = model.copy()
    # rotate the components
    target_map = {'heat': 0, 'hut': 1, 'hot': 2, 'hoot': 3}
    target_labels = (x.split('-')[0] for x in labels[0])
    targets = torch.tensor([target_map[x] for x in target_labels])
    # targets = torch.tensor([[target_map[x] for x in target_labels]] * 800)
    # decoders = [SimpleDecoder(4, len(labels[2]), 1e-4) for _ in range(n_components[0])]
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
    def loss(x: list[torch.Tensor]):
        results = []
        decoders = [SimpleDecoder(4, len(labels[2]), 5e-4) for _ in range(n_components[0])]
        for decoder, xi in zip(decoders, x):
            # decoder = SimpleDecoder(4, xi.shape[1] * xi.shape[2], 5e-3)
            # decoder = decoders[i]
            temp = xi.clone().detach()
            # train the decoder briefly
            Trainer(max_epochs=400,
                    devices=1,
                    accelerator=device,
                    barebones=True,
                    # precision=32,
                    limit_train_batches=1).fit(decoder, (temp, targets))
            y_hat = decoder(xi)
            results.append(decoder.criterion(y_hat, targets))

        results.sort()
        print([r.item() for r in results])
        return results[0]
    model_rot = within_invariance(model_rot, loss, maximize=False, max_iter=100, ignore=(1,))
    T, W, H = model_rot.get_components(numpy=True)[0]

    # %% plot the components
    colors = colors[:n_components[0]]
    conds = {'aud_ls': (-0.5, 1.5), 'aud_lm': (-0.5, 1.5),
             'go_ls': (-0.5, 1.5),
             'resp': (-1, 1)}
    fig, axs = plt.subplots(1, 4)

    # make a plot for each condition in conds as a subgrid
    for j, (cond, times) in enumerate(conds.items()):
        ax = axs[j]
        start = 200 * j
        end = start + 200
        for i in range(n_components[0]):
            fig = plot_dist(
                # H[i],
                model.construct_single_component(0, i).detach().numpy()[:,
                (W[i] / W.sum(0)) > 0.4, start:end].reshape(-1, 200),
                ax=ax, color=colors[i], mode='std', times=times)
        if j == 0:
            ax.legend()
            ax.set_ylabel("Z-Score (V)")
            ylims = ax.get_ylim()
        elif j == 1:
            ax.set_xlabel("Time(s)")
        ax.set_ylim(ylims)
        ax.set_title(cond)
