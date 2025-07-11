import os
from ieeg.io import get_data
from ieeg.arrays.label import LabeledArray
from analysis.grouping import GroupData
from analysis.data import dataloader
from analysis.decoding.utils import extract
from ieeg.viz.ensemble import plot_dist
from analysis.grouping import group_elecs
from itertools import product
from ieeg.viz.parula import parula_map
import torch
import numpy as np
from functools import reduce
import slicetca
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from functools import partial
from tslearn.metrics import SoftDTWLossPyTorch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
torch.set_float32_matmul_precision("medium")

def load_tensor(array, idx, conds, trial_ax, min_nan=5):
    idx = sorted(idx)
    X = extract(array, conds, trial_ax, idx, min_nan)
    std = float(np.nanstd(X.__array__(), dtype='f8'))
    std_ch = np.nanstd(X.__array__(), (0,2,3,4), dtype='f8')
    # mean = float(np.nanmean(X.__array__(), dtype='f8'))
    combined = reduce(lambda x, y: x.concatenate(y, -1),
                      [X[c] for c in conds])
    if (std_ch < (2 * std)).any():
        combined = combined[std_ch < (2 * std),]
    std = float(np.nanstd(combined.__array__(), dtype='f8'))
    out_tensor = torch.from_numpy(combined.__array__() / std)
    mask = torch.isnan(out_tensor)
    # n_nan = mask.sum(dtype=torch.int64)
    # out_tensor[mask] = torch.normal(mean, std, (n_nan,)).to(
    #     out_tensor.dtype)
    return out_tensor, ~mask, combined.labels

HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    n = int(os.environ['SLURM_ARRAY_TASK_ID'])
    print(n)
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    n = 1

log_dir = os.path.join(os.path.dirname(LAB_root), 'logs', str(n))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
layout = get_data('SentenceRep', root=LAB_root)

conds_all = {"resp": (-1, 1), "aud_ls": (-0.5, 1.5),
                 "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
                 "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
                 "go_jl": (-0.5, 1.5)}

def load_spec(group, conds):
    folder = 'stats_freq_hilbert'
    filemask = os.path.join(layout.root, 'derivatives', folder, 'combined',
                            'mask')
    sigs = LabeledArray.fromfile(filemask)
    AUD, SM, PROD, sig_chans, delay = group_elecs(sigs, sigs.labels[1],
                                                  sigs.labels[0])
    idxs = {'SM': SM, 'AUD': AUD, 'PROD': PROD, 'sig_chans': sig_chans,
            'delay': delay}
    idx = sorted(idxs[group])
    filename = os.path.join(layout.root, 'derivatives', folder, 'combined',
                            'zscore')
    zscores = LabeledArray.fromfile(filename, mmap_mode='r')
    neural_data_tensor, mask, labels = load_tensor(zscores, idx,
                                                   conds, 4, 1)
    return neural_data_tensor, mask, labels, idxs

def load_hg(group, conds):
    sub = GroupData.from_intermediates("SentenceRep", LAB_root, folder='stats')
    idxs = {'SM': sub.SM, 'AUD': sub.AUD, 'PROD': sub.PROD, 'sig_chans': sub.sig_chans,
            'delay': sub.delay}
    idx = sorted(idxs[group])
    neural_data_tensor, labels = dataloader(sub.array, idx, conds)
    neural_data_tensor = neural_data_tensor.swapaxes(0, 1).to(torch.float32)
    mask = ~torch.isnan(neural_data_tensor)

    return neural_data_tensor, mask, labels, idxs

# %% grid search
pick_k = True
if pick_k:
    if __name__ == '__main__':
        freeze_support()
    param_grid = {'lr': [1e-4],
                'ranks': [{'min': [1, 0], 'max': [10, 0]},],
                            # {'min': [1], 'max': [10]},],
                  'groups': ['AUD', 'SM', 'PROD', 'sig_chans'],
                  'loss': ['HuberLoss'],
                  'decay': [1],
                  'batch': [False, True],
                  'spec': [False]}
    procs = 1
    threads = 1
    repeats = 10
    conds = ['aud_ls', 'go_ls']
    aud_slice = slice(0, 175)

    for lr, ranks, group, loss, decay, batched, spec in product(
            param_grid['lr'], param_grid['ranks'], param_grid['groups'],
            param_grid['loss'], param_grid['decay'],
    param_grid['batch'], param_grid['spec']):
        if n > 1:
            n -= 1
            continue
        elif n < 1:
            break
        else:
            n -= 1
            print(ranks, group, loss, lr, decay, batched, spec)

        rank_min = ranks['min']
        rank_max = ranks['max']
        if spec:
            neural_data_tensor, mask, labels, idxs = load_spec(group, conds)
            trial_ax = 2
            train_blocks_dimensions = (1, 10, 10)  # Note that the blocks will be of size 2*train_blocks_dimensions + 1
            test_blocks_dimensions = (1, 5, 5)  # Same, 2*test_blocks_dimensions + 1
            if len(ranks['min']) > 1:
                rank_min = ranks['min'] + [0]
                rank_max = ranks['max'] + [0]

        else:
            neural_data_tensor, mask, labels, idxs = load_hg(group, conds)
            trial_ax = 1
            train_blocks_dimensions = (1, 10)
            test_blocks_dimensions = (1, 5)

        idx = sorted(idxs[group])

        kwargs = {'regularization': 'L2'}
        if batched:
            kwargs['batch_dim'] = trial_ax
            kwargs['shuffle_dim'] = 0
            kwargs['precision'] = '16-mixed'
            neural_data_tensor = neural_data_tensor.to(torch.float16)
        else:
            neural_data_tensor = neural_data_tensor.nanmean(trial_ax, dtype=torch.float32)

        ## set up the model
        if not batched:
            train_mask, test_mask = slicetca.block_mask(dimensions=neural_data_tensor.shape,
                                                        train_blocks_dimensions=train_blocks_dimensions, # Note that the blocks will be of size 2*train_blocks_dimensions + 1
                                                        test_blocks_dimensions=test_blocks_dimensions, # Same, 2*test_blocks_dimensions + 1
                                                        fraction_test=0.2)
            # test_mask = torch.logical_and(test_mask, mask)
            # train_mask = torch.logical_and(train_mask, mask)
        else:
            train_mask = mask
            test_mask = None

        loss_grid, seed_grid = slicetca.grid_search(neural_data_tensor,
                                                    min_ranks = rank_min,
                                                    max_ranks = rank_max,
                                                    sample_size=repeats,
                                                    mask_train=train_mask,
                                                    mask_test=test_mask,
                                                    processes_grid=procs,
                                                    processes_sample=threads,
                                                    seed=3,
                                                    batch_prop=decay,
                                                    batch_prop_decay=3 if decay < 1 else 1,
                                                    # min_std=1e-4,
                                                    # iter_std=10,
                                                    init_bias=0.01,
                                                    weight_decay=partial(
                                                        torch.optim.Adam,
                                                        # betas=(0.5, 0.5),
                                                        # amsgrad=True,
                                                        eps=1e-10,
                                                        # weight_decay=0
                                                        ),
                                                    initialization='uniform-positive',
                                                    learning_rate=lr,
                                                    max_iter=1000000,
                                                    positive=True,
                                                    verbose=0,
                                                    loss_function=getattr(torch.nn, loss)(reduction='mean'),
                                                    compile=True,
                                                    min_iter=1,
                                                    gradient_clip_val=1,
                                                    default_root_dir=log_dir,
                                                    dtype=torch.float32,
                                                    # fast_dev_run=True,
                                                    **kwargs)

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
        file_id += "_batched" if batched else ""
        file_id += "_spec" if spec else "_HG"
        file_id += f"_{len(rank_min)}ranks"
        file_id += f"_{loss}"
        file_id += f"_{lr}"
        file_id += f"_{decay}"
        plt.savefig(f"loss_dist_{file_id}.png")
        with open(f"results_grid_{file_id}.pkl", 'wb') as f:
            pickle.dump({'loss': loss_grid.tolist(), 'seed': seed_grid.tolist()}, f)


# %% decompose
decompose = False
if decompose:
    # folder = 'stats_freq_hilbert'
    # filemask = os.path.join(layout.root, 'derivatives', folder, 'combined',
    #                         'mask')
    # sigs = LabeledArray.fromfile(filemask)
    # AUD, SM, PROD, sig_chans, delay = group_elecs(sigs, sigs.labels[1],
    #                                               sigs.labels[0])
    # idxs = {'SM': SM, 'AUD': AUD, 'PROD': PROD, 'sig_chans': sig_chans,
    #         'delay': delay}
    # filename = os.path.join(layout.root, 'derivatives', folder, 'combined',
    #                         'zscore')
    # zscores = LabeledArray.fromfile(filename, mmap_mode='r')
    conds = ['aud_ls', 'go_ls']
    # idx_name = 'SM'
    # with open(r'C:\Users\ae166\Downloads\results2\results_grid_'
    #           f'{idx_name}_3ranks_test_L1Loss_0.0001_1.pkl',
    #           'rb') as f:
    #     results = pickle.load(f)
    # loss_grid = np.array(results['loss']).squeeze()
    # seed_grid = np.array(results['seed']).squeeze()
    # n_components = (np.unravel_index(np.argmin(loss_grid), loss_grid.shape))[0] + 1
    n_components = 5
    # best_seed = seed_grid[
    #     n_components - 1, np.argmin(loss_grid[n_components - 1])]
    best_seed = None
    n_components = (n_components,)
    # neural_data_tensor, mask, labels, idxs = load_hg('SM', conds)
    # neural_data_tensor, mask, labels = load_tensor(zscores, SM, conds, 4, 1)
    neural_data_tensor, mask, labels, idxs = load_spec('SM', conds)
    # neural_data_jl, _, _ = load_tensor(zscores, SM, conds_jl, 4, 1)
    # trial_av = neural_data_tensor.nanmean(2, dtype=torch.float32)
    # trial_av.to('cuda')
    # idx_name = 'sig_chans'
    # neural_data_tensor = neural_data_tensor.to(torch.bfloat16)

    n = 0
    # %%
    losses, model = slicetca.decompose(
        # trial_av,
        neural_data_tensor,
        # n_components,
        (n_components[0], 0, 0),
        seed=best_seed,
        positive=True,
        # min_std=9e-3,
        # iter_std=20,
        learning_rate=5e-3,
        max_iter=1000,
        batch_dim=2,
        batch_prop=1,
        batch_prop_decay=3,
        # weight_decay=partial(torch.optim.RMSprop,
        #                      eps=1e-9,
        #                      momentum=0.9,
        #                      # alpha=0.5,
        #                      centered=True,
        #                      weight_decay=1e-4),
        # weight_decay=partial(torch.optim.Rprop),#, etas=(0.5, 1.2), step_sizes=(1e-8, 1)),
        weight_decay=partial(torch.optim.Adam,
                                # betas=(0.5, 0.5),
                                # amsgrad=True,
                                # eps=1e-9,
                                # weight_decay=1e-6
                             ),
        # weight_decay=partial(torch.optim.LBFGS, max_eval=200,
        #                      tolerance_grad=1e-6,
        #                      line_search_fn='strong_wolfe'),
        mask=mask,
        init_bias=0.1,
        initialization='uniform-positive',
        # loss_function=torch.nn.HuberLoss(
        #     reduction='mean'),
        loss_function=SoftDTWLossPyTorch(1, True),#, _euclidean_squared_dist),
        # loss_function=partial(soft_dtw_normalized, gamma=1.0, normalize=True),
        verbose=0,
        compile=True,
        shuffle_dim=0,
        device='cuda',
        # default_root_dir=os.path.join(os.path.dirname(LAB_root), 'logs'),
        gradient_clip_val=1,
        accumulate_grad_batches=3,
        # reload_dataloaders_every_n_epochs=1,
        regularization='L2',
        min_iter=10,
        precision='16-mixed',
        dtype=torch.float32,
        testing=False,
    )
    # torch.save(model, f'model_{'SM'}_freq.pt')

    # plot the losses
    plt.figure(figsize=(4, 3), dpi=100)
    plt.plot(np.arange(100, len(model.losses)), model.losses[100:], 'k')
    plt.xlabel('iterations')
    plt.ylabel('mean squared error')
    plt.xlim(0, len(model.losses))
    plt.tight_layout()
    # %% plot the model
    idx1 = np.linspace(0, labels[0].shape[0], 8).astype(int)[1:-1]
    idx2 = np.linspace(0, labels[1].shape[0], 6).astype(int)[1:-1]
    timings = {'aud_ls': range(0, 200),
               'go_ls': range(200, 400),}
               # 'go_lm': range(800, 1000)}
    components = model.get_components(numpy=True)
    figs = {}
    for cond, timing in timings.items():
        comp = model.get_components(numpy=True)
        # comp[n] = [comp[n][1][..., timing]]
        comp[n][1] = comp[n][1][..., timing]
        # comp[n][0] = np.array([])
        if cond.startswith('aud_l'):
            t_label = f"Time (s) from Stimulus"
        elif cond.startswith('go_lm'):
            t_label = f"Time (s) from Go Cue (Mime)"
        elif cond.startswith('go_ls'):
            t_label = f"Time (s) from Go Cue (Speak)"
        elif cond.startswith('go_jl'):
            t_label = f"Time (s) from Go Cue (:=:)"
        axes = slicetca.plot(model,
                             components=comp,
                             ignore_component=(0,),
                             variables=('channel',
                                        # 'freq',
                                        t_label),
                             sorting_indices=(None,
                                              # labels[1].astype(float).argsort()[::-1],
                                              None),
                             ticks=(None,
                                    # idx2[::-1],
                                    [0, 49, 99, 149, 199]),
                             tick_labels=(labels[0][idx1],
                                          # labels[1][idx2].astype(float).astype(int),
                                          [-0.5, 0, 0.5, 1, 1.5]),
                             cmap=parula_map)
    colors = ['orange', 'y', 'k', 'c', 'm', 'deeppink',
              'darkorange', 'lime', 'blue', 'red', 'purple']
    # %% plot the sensory motors
    timingss = [{'aud_ls': range(0, 200),
               'go_ls': range(600, 800)},
               {'aud_jl': range(400, 600),
                'go_jl': range(1000, 1200)}]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    ylims = [0, 0]
    lss = ['-', '--']
    for i, timings in enumerate(timingss):
        for j, (cond, time_slice) in enumerate(timings.items()):
            ax = axs[j]
            mat = np.nanmean(zscores[cond][:,sorted(SM)].__array__(), axis=(0,2,3))
            plot_dist(mat,#(trial_av.mean(1)*2).detach().cpu().numpy()[..., time_slice],
                            ax=ax, color='red', linestyle=lss[i], label=cond[-2:],
                      times=(-0.5, 1.5))
            if cond.startswith('go'):
                event = "Go Cue"
            elif cond.startswith('aud'):
                event = "Stimulus"
            if i == 0:
                if j == 0:
                    # ax.legend()
                    ax.set_ylabel("Z-Score (V)")
            elif i == 1:
                if j == 0:
                    ax.legend(loc='best')

                ax.set_xlabel("Time(s) from " + event)
            ylim = ax.get_ylim()
            ylims[1] = max(ylim[1], ylims[1])
            ylims[0] = min(ylim[0], ylims[0])
    for ax in axs:
        ax.set_ylim(ylims)
        ax.axhline(0, color='k', linestyle='--')
    fig.suptitle('Sensory Motor')

    # %% plot the components
    W, H = model.get_components(numpy=True)[n]
    timings = {'aud_ls': range(0, 200),
               'go_ls': range(200, 400)}
    idx_name = 'SM'
    colors = colors[:n_components[n]]
    conds = {'aud_ls': (-0.5, 1.5),
             'go_ls': (-0.5, 1.5)}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    ylims = [0, 0]
    # make a plot for each condition in conds as a subgrid
    for j, (cond, times) in enumerate(conds.items()):
        ax = axs[j]
        for i in range(n_components[n]):
            fig = plot_dist(
                # H[i],
                model.construct_single_component(n, i).to(torch.float32).detach().cpu().numpy()[
                (W[i] / W.sum(0)) > 0.3][
                    ..., timings[cond]].reshape(-1, 200),
                ax=ax, color=colors[i], mode='sem', times=times,
                label=f"Component {colors[i]}")


        if cond.startswith('go'):
            event = "Go Cue"
        elif cond.startswith('aud'):
            event = "Stimulus"
        if j == 0:
            # ax.legend()
            ax.set_ylabel("Z-Score (V)")

        ax.set_xlabel("Time(s) from " + event)
        ylim = ax.get_ylim()
        ylims[1] = max(ylim[1], ylims[1])
        ylims[0] = min(ylim[0], ylims[0])
        # ax.set_title(cond)
    for ax in axs:
        ax.set_ylim(ylims)
    plt.suptitle(f"Components")

    # %% plot the components
    W, H = model.get_components(numpy=True)[n]
    timingss = [{'aud-Listen-Speak': range(0, 200),
               'go-Listen-Speak': range(600, 800)},
               {'aud_jl': range(400, 600),
                'go-Just-Listen': range(1000, 1200)}]
    idx_name = 'SM'
    colors = colors[:n_components[n]]
    # conds = {'aud_ls': (-0.5, 1.5),
    #          'go_ls': (-0.5, 1.5)}
    fig, axs = plt.subplots(n_components[n], 2, dpi=100)
    ylims = [[], []]
    # make a plot for each condition in conds as a subgrid
    for i, timings in enumerate(timingss):
        for j, (cond, times) in enumerate(timings.items()):
            for k in range(n_components[n]):
                ax = axs[k, j]
                if i == 0:
                    ls = '-'
                else:
                    ls = '--'
                plot_dist(
                    # H[k],
                    model.construct_single_component(n, k).detach().cpu().numpy()[
                    (W[k] / W.sum(0)) > 0.5][
                        ..., times].reshape(-1, 200),
                    ax=ax, color=colors[k], mode='sem', times=(-0.5, 1.5),
                    linestyle=ls, label=cond[3:])
                ylim = ax.get_ylim()
                ylims[1].append(ylim[1])
                ylims[0].append(ylim[0])
                if j == 1 and i == 1:
                    ax.legend()

            if cond.startswith('go'):
                event = "Go Cue"
            elif cond.startswith('aud'):
                event = "Stimulus"

            #     ax.set_ylabel("Z-Score (V)")

            ax.set_xlabel("Time(s) from " + event)

    for ax in axs.flat:
        ax.set_ylim((min(ylims[0]), max(ylims[1])))
    # plt.suptitle(f"Components")
    fig.supylabel('Z-Score (V)')

    # %% plot the region membership
    from ieeg.viz.mri import electrode_gradient, plot_on_average

    chans = ['-'.join([f"D{int(ch.split('-')[0][1:])}", ch.split('-')[1]]) for
             ch in labels[0]]
    subj = sorted(set(ch.split('-')[0] for ch in chans))
    electrode_gradient(layout.get_subjects(), W*2, chans, colors, mode='both', fig_dims=(W.shape[0],1))

    # %%
    from ieeg.viz.mri import gen_labels, subject_to_info, Atlas
    import matplotlib.pyplot as plt

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
        sm_elecs = zscores.labels[2][idx]
        groups = {r: [] for r in rois}
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
