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
from multiprocessing import freeze_support
import pickle

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
torch.set_float32_matmul_precision("medium")

def load_tensor(array, idx, conds, trial_ax):
    idx = sorted(idx)
    X = extract(array, conds, trial_ax, idx)
    std = float(np.nanstd(X.__array__(), dtype='f8'))
    std_ch = np.nanstd(X.__array__(), (0,2,3,4), dtype='f8')
    mean = float(np.nanmean(X.__array__(), dtype='f8'))
    combined = reduce(lambda x, y: x.concatenate(y, -1),
                      [X[c] for c in conds])
    if (std_ch < (2 * std)).any():
        combined = combined[std_ch < (2 * std),]
        std = float(np.nanstd(combined.__array__(), dtype='f8'))
    out_tensor = torch.from_numpy((combined.__array__() / std))
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
folder = 'stats_freq_multitaper_std'
loader = DataLoader(layout, conds_all, 'significance', True, folder,
                   '.h5')
filemask = os.path.join(layout.root, 'derivatives', folder, 'combined', 'mask')
if not os.path.exists(filemask + ".npy"):
    sigs = LabeledArray.from_dict(combine(loader.load_dict(
        dtype=bool, n_jobs=-1), (0, 2)), dtype=bool)
    sigs.tofile(filemask)
else:
    sigs = LabeledArray.fromfile(filemask)
AUD, SM, PROD, sig_chans = group_elecs(sigs, sigs.labels[1], sigs.labels[0])
idxs = {'SM': SM, 'AUD': AUD, 'PROD': PROD, 'sig_chans': sig_chans}

filename = os.path.join(layout.root, 'derivatives', folder, 'combined', 'zscores')
zscores = LabeledArray.fromfile(filename, mmap_mode='r')
conds = ['aud_ls', 'aud_lm', 'go_ls', 'go_lm', 'resp']

# %% grid search
pick_k = False
if pick_k:
    if __name__ == '__main__':
        freeze_support()
    param_grid = {'ranks': [#{'min': [1, 0, 0], 'max': [12, 0, 0]},
                            {'min': [1], 'max': [12]},],
                  'groups': ['AUD', 'SM', 'PROD', 'sig_chans'],
                  'masks': [
                      {'train': False, 'test': False},
                      {'train': True, 'test': True},
                            ],
                  'loss': ['HuberLoss', 'L1Loss'],
                  'lr': [1e-3],
                  'decay': [1]
                    }
    procs = 1
    threads = 1
    repeats = 2
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
    # idx_name = 'SM'
    # with open(r'C:\Users\ae166\Downloads\results2\results_grid_'
    #           f'{idx_name}_3ranks_test_L1Loss_0.0001_1.pkl',
    #           'rb') as f:
    #     results = pickle.load(f)
    # loss_grid = np.array(results['loss']).squeeze()
    # seed_grid = np.array(results['seed']).squeeze()
    # n_components = (np.unravel_index(np.argmin(loss_grid), loss_grid.shape))[0] + 1
    n_components = 6
    # best_seed = seed_grid[
    #     n_components - 1, np.argmin(loss_grid[n_components - 1])]
    best_seed = None
    n_components = (n_components,)
    neural_data_tensor, mask, labels = load_tensor(zscores, sig_chans, conds, 4)
    trial_av = neural_data_tensor.to(torch.float32).nanmean(2)
    # trial_av.to('cuda')
    idx_name = 'sig_chans'
    # trial_av = neural_data_tensor.to(torch.float32)
    n = 0
    # %%
    losses, model = slicetca.decompose(
        trial_av,
        # neural_data_tensor,

                                       # n_components,
                                       (n_components[0], 0, 0),
                                       seed=best_seed,
                                       positive=True,
                                       min_std=5e-5,
                                       iter_std=1000,
                                       learning_rate=1e-4,
                                       max_iter=1000000,
                                       # batch_dim=2,
                                       # batch_prop=0.1,
                                       # batch_prop_decay=5,
                                       weight_decay=0.1,
                                       # mask=mask,
                                       init_bias=0.01,
                                       initialization='uniform-positive',
                                       loss_function=torch.nn.L1Loss(
                                           reduction='mean'),
                                       verbose=0,
                                       compile=True,
    device='cuda')
    # torch.save(model, f'model_{idx_name}_freq.pt')

    # %% plot the losses
    plt.figure(figsize=(4, 3), dpi=100)
    plt.plot(np.arange(100, len(model.losses)), model.losses[100:], 'k')
    plt.xlabel('iterations')
    plt.ylabel('mean squared error')
    plt.xlim(0, len(model.losses))
    plt.tight_layout()
    # %% plot the model
    idx1 = np.linspace(0, labels[0].shape[0], 8).astype(int)[1:-1]
    idx2 = np.linspace(0, labels[1].shape[0], 6).astype(int)[1:-1]
    axes = slicetca.plot(model,
                         variables=('channel', 'freq', 'time'),
                         ticks=(None, idx2, None),
                         tick_labels=(labels[0][idx1], labels[1][idx2].astype(float).astype(int), labels[3]))
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
                (W[i] / W.sum(0)) > 0.5][
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

    # %% plot the region membership
    from ieeg.viz.mri import gen_labels, subject_to_info, Atlas

    # colors = ['Late Prod', 'WM', 'Feedback', 'Instructional', 'Early Prod']
    rois = ['IFG', 'Tha', 'PoG', 'Amyg', 'PhG', 'MVOcC', 'ITG', 'PrG', 'PCL',
            'IPL', 'MFG', 'CG', 'Pcun', 'BG',
            'INS', 'FuG', 'LOcC', 'STG', 'OrG', 'MTG', 'pSTS', 'Hipp', 'SFG',
            'SPL']
    names = ['orange', 'yellow', 'black', 'cyan', 'magenta', 'deeppink',
             'darkorange', 'lime', 'blue', 'red', 'purple']
    atlas = Atlas()
    fig, axs = plt.subplots(n_components[0], 1)
    split = (l.split('-') for l in labels[0])
    lzfilled = (f"D{s[1:].zfill(4)}-{ch}" for s, ch in split)
    sm_idx = [zscores.labels[2].tolist().index(l) for l in lzfilled]
    idxs = [torch.tensor(sm_idx)[
                (W[i] / W.sum(0)) > 0.4
                # W.argmax(0) == i
                ].tolist() for i in range(n_components[0])]
    ylims = [0, 0]
    all_groups = []
    for idx in idxs:
        groups = {r: [] for r in rois}
        sm_elecs = zscores.labels[2][idx]
        for subj in layout.get_subjects():
            subj_old = f"D{int(subj[1:])}"
            info = subject_to_info(subj_old)
            ch_labels = gen_labels(info, subj_old, atlas='.BN_atlas')
            for key, value in ch_labels.items():
                item = subj + '-' + key
                if item in sm_elecs:
                    roi = value.split("_")[0]
                    try:
                        area = atlas[value.split("_")[0]].gyrus
                    except KeyError as e:
                        if value.split("_")[0] == 'TE1.0/TE1.2':
                            area = 'STG'
                        else:
                            print(e)
                            continue
                    if area in groups.keys():
                        groups[area].append(subj + '-' + key)
        groups_num = {key: len(value) for key, value in groups.items()}
        all_groups.append(groups_num)
        # plot the histogram, with

    filtered_groups_num = [{} for _ in range(n_components[0])]
    for roi in rois:
        # new_roi = atlas.abbreviations[roi]
        new_roi = roi
        if any(group[roi] > 0 for group in all_groups):
            for i, group in enumerate(all_groups):
                filtered_groups_num[i][new_roi] = group[roi]
    for i, (c, ax) in enumerate(zip(names, axs)):
        ax.bar(filtered_groups_num[i].keys(), filtered_groups_num[i].values())
        plt.sca(ax)
        # if ax is axs[-1]:
        if True:
            plt.xticks(rotation=45)
        else:
            plt.xticks([])
        ax.set_ylabel(f"{c}")
        ylim = ax.get_ylim()
        ylims[1] = max(ylim[1], ylims[1])
        ylims[0] = min(ylim[0], ylims[0])
    for ax in axs:
        ax.set_ylim(ylims)

    # %% plot each component
    n_comp = W.shape[0]
    fig, axs = plt.subplots(2*len(timings), n_comp)
    data = neural_data_tensor.mean((1, 2)).detach().cpu().numpy()
    for j, (cond, times) in enumerate(timings.items()):
        j *= 2
        ylims = [0, 0]
        for i, ax in enumerate(axs[0 + j]):
            trimmed = data[(W[i] / W.sum(0)) > 0.4][:, times]
            sorted_trimmed = trimmed[
                                 np.argsort(W[i, (W[i] / W.sum(0)) > 0.4])][
                             ::-1]
            plot_dist(trimmed.reshape(-1, 200), ax=ax, color=colors[i],
                      mode='sem', times=conds[cond])
            ax.set_xticks([])
            ax.set_xticklabels([])
            ylims[1] = max(ax.get_ylim()[1], ylims[1])
            ylims[0] = min(ax.get_ylim()[0], ylims[0])

            scale = np.mean(sorted_trimmed) + 2 * np.std(sorted_trimmed)
            axs[1 + j, i].imshow(sorted_trimmed, aspect='auto', cmap='inferno',
                                 vmin=0, vmax=scale,
                                 extent=[conds[cond][0], conds[cond][-1], 0,
                                         len(sorted_trimmed)])

            if i > 0:
                axs[0 + j, i].set_yticks([])
                axs[0 + j, i].set_yticklabels([])
                axs[1 + j, i].set_yticks([])
                axs[1 + j, i].set_yticklabels([])
            else:
                axs[0 + j, i].set_ylabel("Z-Score (V)")
                axs[1 + j, i].set_ylabel("Channels")
            if i == n_components[n] // 2:
                axs[0 + j, i].set_title(f"{cond}")

        for ax in axs[0 + j]:
            ax.set_ylim(ylims)
        # fig.suptitle(cond)
        fig.tight_layout()

    # # %% plot the components
    # colors = colors[:n_components[0]]
    # conds = {'aud_ls': (-0.5, 1.5),
    #          'go_ls': (-0.5, 1.5),
    #          'resp': (-1, 1)}
    # fig, axs = plt.subplots(1, 3)
    #
    # # make a plot for each condition in conds as a subgrid
    # for j, (cond, times) in enumerate(conds.items()):
    #     ax = axs[j]
    #     start = 200 * j
    #     end = start + 200
    #     for i in range(n_components[0]):
    #         fig = plot_dist(
    #             # H[i],
    #             model.construct_single_component(0, i).detach().numpy()[
    #             (W[i] / W.sum(0)) > 0.3, ..., start:end].reshape(-1, 200),
    #             ax=ax, color=colors[i], mode='sem', times=times)
    #     if j == 0:
    #         ax.legend()
    #         ax.set_ylabel("Z-Score (V)")
    #         ylims = ax.get_ylim()
    #     elif j == 1:
    #         ax.set_xlabel("Time(s)")
    #     ax.set_ylim(ylims)
    #     ax.set_title(cond)
