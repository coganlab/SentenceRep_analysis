import os
from ieeg.io import get_data, DataLoader
from ieeg.arrays.label import LabeledArray, combine
from ieeg.decoding.decode import extract
from ieeg.viz.ensemble import plot_dist
from analysis.grouping import group_elecs
from itertools import product
import torch
import numpy as np
from functools import reduce
import slicetca
import matplotlib.pyplot as plt


def load_tensor(array, idx, conds, trial_ax):
    idx = sorted(idx)
    X = extract(array, conds, trial_ax, idx)
    std = float(np.nanstd(X.__array__(), dtype='f8'))
    mean = float(np.nanmean(X.__array__(), dtype='f8'))
    combined = reduce(lambda x, y: x.concatenate(y, -1),
                      [X[c] for c in conds])
    out_tensor = torch.from_numpy(
        (combined.__array__() / std))
    mask = torch.isnan(out_tensor)
    n_nan = mask.sum(dtype=torch.int64)
    out_tensor[mask] = torch.normal(mean, std, (n_nan,)).to(
        out_tensor.dtype)
    return out_tensor, mask, combined.labels

HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    n = int(os.environ['SLURM_ARRAY_TASK_ID'])
    print(n)
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    n = 1

layout = get_data('SentenceRep', root=LAB_root)

conds_all = {"resp": (-1, 1), "aud_ls": (-0.5, 1.5),
                 "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
                 "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
                 "go_jl": (-0.5, 1.5)}
loader = DataLoader(layout, conds_all, 'significance', True, 'stats_freq',
                   '.h5')
sigs = LabeledArray.from_dict(combine(loader.load_dict(
    dtype=bool, n_jobs=-1), (0, 2)), dtype=bool)
filename = os.path.join(layout.root, 'derivatives', 'stats_freq', 'combined', 'zscores')
zscores = LabeledArray.fromfile(filename, mmap_mode='r')
AUD, SM, PROD, sig_chans = group_elecs(sigs, sigs.labels[1], sigs.labels[0])
idxs = {'SM': SM, 'AUD': AUD, 'PROD': PROD, 'sig_chans': sig_chans}
ch_names = sigs.labels[1]
conds = ['aud_ls', 'aud_lm', 'go_ls', 'go_lm', 'resp']
idx_name = 'Sensory-Motor'

# %% grid search
pick_k = False
if pick_k:
    param_grid = {'ranks': [{'min': [1, 0, 0], 'max': [12, 0, 0]},
                            {'min': [1], 'max': [12]},],
                  'groups': ['AUD', 'SM', 'PROD', 'sig_chans'],
                  'masks': [
                      {'train': False, 'test': False},
                      {'train': True, 'test': True},
                            ],
                  'loss': ['HuberLoss', 'L1Loss'],
                  'lr': [1e-3, 1e-4],
                  'decay': [1]
                    }
    procs = 1
    threads = 4
    repeats = 4
    conds = ['aud_ls', 'aud_lm', 'go_ls', 'go_lm', 'resp']
    aud_slice = slice(0, 175)

    for ranks, group, is_mask, loss, lr, decay in product(
            param_grid['ranks'], param_grid['groups'], param_grid['masks'],
            param_grid['loss'], param_grid['lr'], param_grid['decay']):
        if n > 1:
            n -= 1
            continue
        else:
            print(ranks, group, is_mask, loss, lr, decay)
        idx = sorted(idxs[group])
        neural_data_tensor, mask, labels = load_tensor(zscores, idx,
                                                       conds, 4)
        mask = mask.any(2)
        trial_av = neural_data_tensor.to(torch.float32).nanmean(2)

        ## set up the model
        if is_mask['train'] and is_mask['test']:
            train_mask, test_mask = slicetca.block_mask(dimensions=trial_av.shape,
                                                        train_blocks_dimensions=(1, 10, 10), # Note that the blocks will be of size 2*train_blocks_dimensions + 1
                                                        test_blocks_dimensions=(1, 5, 5), # Same, 2*test_blocks_dimensions + 1
                                                        fraction_test=0.2)
            test_mask = torch.logical_and(test_mask, mask)
            train_mask = torch.logical_and(train_mask, mask)
        else:
            train_mask = mask
            test_mask = None

        loss_grid, seed_grid = slicetca.grid_search(trial_av,
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
                                                    # weight_decay=decay,
                                                    initialization='uniform-positive',
                                                    learning_rate=lr,
                                                    max_iter=1000000,
                                                    positive=True,
                                                    verbose=0,
                                                    # batch_dim=0,
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


# %% decompose
decompose = True
if decompose:
    neural_data_tensor, mask, labels = load_tensor(zscores, sig_chans, conds,
                                                   4)
    trial_av = neural_data_tensor.to(torch.float32).nanmean(2)
    n_components = [5]
    n = 0
    losses, model = slicetca.decompose(trial_av,#.to(torch.float32),
                                       # n_components,
                                       (n_components[0], 0, 0),
                                       seed=None,
                                       positive=True,
                                       # min_std=5e-5,
                                       # iter_std=1000,
                                       learning_rate=5e-4,
                                       max_iter=1000000,
                                       # batch_dim=0,
                                       # batch_prop=1,
                                       # batch_prop_decay=5,
                                       weight_decay=0.33,
                                       # mask=mask,
                                       init_bias=0.01,
                                       initialization='uniform-positive',
                                       loss_function=torch.nn.HuberLoss(
                                           reduction='mean'),
                                       verbose=0
                                       )

    # %% plot the losses
    plt.figure(figsize=(4, 3), dpi=100)
    plt.plot(np.arange(1000, len(model.losses)), model.losses[1000:], 'k')
    plt.xlabel('iterations')
    plt.ylabel('mean squared error')
    plt.xlim(0, len(model.losses))
    plt.tight_layout()
    # %% plot the model
    axes = slicetca.plot(model,
                         variables=('channel', 'freq', 'time'), )
    colors = ['orange', 'y', 'k', 'c', 'm', 'deeppink',
              'darkorange', 'lime', 'blue', 'red', 'purple']
    W, H = model.get_components(numpy=True)[n]
    # %% plot the components
    colors = colors[:n_components[n]]
    conds = {'aud_ls': (-0.5, 1.5),
             'go_ls': (-0.5, 1.5),
             'resp': (-1, 1)}
    timings = {'aud_ls': range(0, 200),
               'go_ls': range(400, 600),
               'resp': range(800, 1000)}
    fig, axs = plt.subplots(1, 3)
    ylims = [0, 0]
    # make a plot for each condition in conds as a subgrid
    for j, (cond, times) in enumerate(conds.items()):
        ax = axs[j]
        for i in range(n_components[n]):
            fig = plot_dist(
                # H[i],
                model.construct_single_component(n, i).detach().cpu().numpy()[
                (W[i] / W.sum(0)) > 0.4][
                    ..., timings[cond]].reshape(-1, 200),
                ax=ax, color=colors[i], mode='sem', times=times,
                label=f"Component {colors[i]}")
        if j == 0:
            # ax.legend()
            ax.set_ylabel("Z-Score (V)")

        elif j == 1:
            ax.set_xlabel("Time(s)")
        ylim = ax.get_ylim()
        ylims[1] = max(ylim[1], ylims[1])
        ylims[0] = min(ylim[0], ylims[0])
        ax.set_title(cond)
    for ax in axs:
        ax.set_ylim(ylims)
    plt.suptitle(f"{idx_name}")
