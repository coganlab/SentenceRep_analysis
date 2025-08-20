# %%
import os
from ieeg.io import get_data
from ieeg.arrays.label import LabeledArray
from analysis.grouping import GroupData
from analysis.data import dataloader
from analysis.decoding.utils import extract
from ieeg.viz.ensemble import plot_dist
from analysis.grouping import group_elecs
import torch
import numpy as np
from functools import reduce
import slicetca
import matplotlib.pyplot as plt
from functools import partial
from slicetca.run.dtw import SoftDTW
import pickle

SoftDTW.__module__ = "tslearn.metrics"

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
torch.set_float32_matmul_precision("medium")

exclude = ["D0063-RAT1", "D0063-RAT2", "D0063-RAT3", "D0063-RAT4",
           "D0053-LPIF10", "D0053-LPIF11", "D0053-LPIF12", "D0053-LPIF13",
           "D0053-LPIF14", "D0053-LPIF15", "D0053-LPIF16",
           "D0027-LPIF6", "D0027-LPIF7", "D0027-LPIF8", "D0027-LPIF9",
           "D0027-LPIF10"]

def load_tensor(array, idx, conds, trial_ax, min_nan=1):
    idx = sorted(idx)
    if exclude:
        idx_map = {k: v for k, v in enumerate(array.labels[2])}
        idx = [i for i in idx if idx_map[i] not in exclude]

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

def load_spec(group, conds, folder='stats_freq_hilbert'):
    # folder = 'stats_freq_super'
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

def load_hg(group, conds, **kwargs):
    sub = GroupData.from_intermediates("SentenceRep", LAB_root, folder='stats')
    idxs = {'SM': sub.SM, 'AUD': sub.AUD, 'PROD': sub.PROD, 'sig_chans': sub.sig_chans,
            'delay': sub.delay}
    idx = sorted(idxs[group])
    neural_data_tensor, labels = dataloader(sub.array, idx, conds, **kwargs)
    neural_data_tensor = neural_data_tensor.swapaxes(0, 1).to(torch.float32)
    labels[0], labels[1] = labels[1], labels[0]
    mask = ~torch.isnan(neural_data_tensor)

    return neural_data_tensor, mask, labels, idxs

def split_and_stack(tensor, split_dim, stack_pos, num_splits, new_dim: bool = True):
    # Split tensor along split_dim
    splits = torch.split(tensor, tensor.shape[split_dim] // num_splits, dim=split_dim)
    # Stack splits into a new axis
    stacked = torch.stack(splits, dim=0)
    # Move new axis to stack_pos
    permute_order = list(range(stacked.ndim))
    permute_order.insert(stack_pos, permute_order.pop(0))
    out = stacked.permute(permute_order)
    if not new_dim:
        # new_dim is false, combine the new axis with the next axis
        out = out.reshape(*out.shape[:stack_pos], -1, *out.shape[stack_pos + 2:])
    return out

def to_plot(mode, W, H, neural_data, timings, i, cond, j = 0):
    if mode == 'weights':
        data = H[i, j, ..., timings[cond]]
    elif mode == 'components':
        data = model.construct_single_component(0, i).to(torch.float32).detach().cpu().numpy()[
        (W[i] / W.sum(0)) > 0.5][
            :, j, ..., timings[cond]
        ].reshape(-1, 200)
    else:
        data = (neural_data[..., timings[cond]] * W[i, :, None, None, None])[(W[i] / W.sum(0)) > 0.5].nanmean(axis=(1,2)).detach().cpu().numpy()
    return data

def plot_components(model, mode = 'weights', neural_data = None, plot_latencies = True):
    assert mode in ('weights', 'components', 'weighted')
    assert mode in ('weights', 'components') or neural_data is not None
    colors = ['orange', 'y', 'k', 'c', 'm', 'deeppink',
              'darkorange', 'lime', 'blue', 'red', 'purple']
    W, H = model.get_components(numpy=True)[0]
    # W2 = (np.cov(
    #     model.forward().detach().cpu().numpy().reshape(W.shape[1],
    #                                                    -1)) @ W.T).T
    # H2 = H @ np.cov(
    #     model.forward().detach().cpu().numpy().reshape(-1, H.shape[-1]),
    #     rowvar=False)
    timings = {'aud_ls': slice(0, 200),
               'go_ls': slice(200, 400)}
    n_components = W.shape[0]
    # idx_name = 'SM'
    colors = colors[:n_components]
    conds = {'aud_ls': (-0.5, 1.5),
             'go_ls': (-0.5, 1.5)}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    ylims = [0, 0]
    # make a plot for each condition in conds as a subgrid
    for j, (cond, times) in enumerate(conds.items()):
        ax = axs[j]
        for i in range(n_components):
            plot_dist(
                to_plot(mode, W, H, neural_data, timings, i, cond),
                # model.construct_single_component(0, i).to(torch.float32).detach().cpu().numpy()[
                # (W[i] / W.sum(0)) > 0.5][
                #     ..., 0, timings[cond]
                # ].reshape(-1, len(timings[cond])),
                # (all_con[:, 0, :, timings[cond]] * W[i, :, None,
                #                                    None]).nanmean(axis=0).detach().cpu().numpy(),
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
    if not plot_latencies:
        return fig, axs
    for j, (cond, times) in enumerate(conds.items()):
        ax = axs[j]
        ax.set_ylim(ylims)
        positions = np.linspace((ylims[0] + ylims[1])* 4 / 5, ylims[1], n_components)
        width = (positions[1] - positions[0])
        positions -= width / 2
        for i in range(n_components):
            # make a horizontal boxplot of the peak times
            data = to_plot(mode, W, H, neural_data, timings, i, cond)
            ttimes = np.linspace(times[0], times[1], data.shape[-1])
            peak_times = ttimes[data.argmax(axis=-1)]
            ax.boxplot(peak_times, vert=False, manage_ticks=False,
                       positions=[positions[i]], # whis=[15, 85],
                       widths=(positions[1] - positions[0])/2,
                       patch_artist=True, boxprops=dict(facecolor=colors[i]),
                       medianprops=dict(color='k', alpha=0.5), showfliers=False)
    return fig, axs

def plot_data(chn_label: str, labels, data_tensor):
    chn = labels[0].find(chn_label)
    freqs = np.array([float(l) for l in labels[1]])
    data = np.nanmean(data_tensor[chn, np.where(np.logical_and(150 > freqs, freqs > 70)), :, :200].detach().cpu().numpy(), axis=(0, 1))
    # data = neural_data_tensor[chn, :, :200].detach().cpu().numpy()
    ax = plot_dist(data,
                   mode='std', linewidth=4, times=(-0.5, 1.5))
    plot_dist(data.T[None],
              mode='std', ax=ax, linewidth=0.5, times=(-0.5, 1.5))
    fig, axs = plt.subplots(5, 7)
    j = 0
    for i in range(data_tensor.shape[2]):
        idx = i - j
        ax = axs[idx // 7, idx % 7]
        data = data_tensor[chn, :, i, :200].detach().cpu().numpy()
        if np.isnan(data).any():
            j += 1
            continue
        ax.imshow(data, aspect='auto', origin='lower',)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axvline(50, color='k', linestyle='--')
        plot_dist(data,
                  mode='std', linewidth=4, times=(-0.5, 1.5), ax=ax)
        plot_dist(data.T[None],
                  mode='std', ax=ax, linewidth=0.5, times=(-0.5, 1.5))


# %% decompose
decompose = True
if decompose:
    # folder = 'stats_freq_hilbert'
    # filemask = os.path.join(layout.root, 'derivatives', folder, 'combined',
    #                         'mask')
    # sigs = LabeledArray.fromfile(filemask)
    # idxs = {'SM': SM, 'AUD': AUD, 'PROD': PROD, 'sig_chans': sig_chans,
    #         'delay': delay}
    # filename = os.path.join(layout.root, 'derivatives', folder, 'combined',
    #                         'zscore')
    # zscores = LabeledArray.fromfile(filename, mmap_mode='r')
    conds = ['aud_ls', 'go_ls', 'aud_lm', 'go_lm', 'aud_jl', 'go_jl']
    idx_name = 'PROD'
    with open(r'C:\Users\ae166\Downloads\8-19-25\results_'
              f'{idx_name}_unbatched_spec_4ranks_L1Loss_0.001_1.pkl',
              'rb') as f:
        results = pickle.load(f)
    loss_grid = np.array(list(results.values()))
    seed_grid = np.array(list(results.keys()))
    # n_components = (np.unravel_index(np.argmin(loss_grid), loss_grid.shape))[0] + 1
    n_components = 5
    best_seed = seed_grid[np.argmin(loss_grid[seed_grid[:,0] == n_components]), -1]
    # best_seed = None
    n_components = (n_components,)
    # _, _, labels, idxs = load_hg('SM', conds)

    # neural_data_tensor, mask, labels = load_tensor(zscores, SM, conds, 4, 1)
    neural_data_tensor, mask, labels, idxs = load_spec(idx_name, conds, 'stats_freq_hilbert')
    # plot_data('D0005-PST3', labels, neural_data_tensor)
    # neural_data_jl, _, _ = load_tensor(zscores, SM, conds_jl, 4, 1)
    # trial_av = neural_data_tensor.nanmean(1, dtype=torch.float32)
    all_con = split_and_stack(neural_data_tensor, -1, 1, 3)
    # all_con = torch.cat([all_con[..., :150], all_con[..., 200:]], dim=-1)
    all_mask = split_and_stack(mask, -1, 1, 3)
    # all_mask = torch.cat([all_mask[..., :150], all_mask[..., 200:]], dim=-1)
    trial_av = torch.zeros(all_con.shape[:-2] + (all_con.shape[-1],), dtype=torch.float32)
    for i in range(all_con.shape[0]):
        trial_av[i] = all_con[i].nanmean(-2, dtype=torch.float32)
    # trial_av = all_con.nanmean(-2, dtype=torch.float32)
    # from tslearn.metrics import gamma_soft_dtw
    # gammas = torch.tensor([
    #     gamma_soft_dtw(torch.nanmean(all_con[i, 0], 1).detach().cpu().numpy(), 400)
    #     for i in range(all_con.shape[0])
    # ])
    # neural_data_tensor = neural_data_tensor.to(torch.bfloat16)

    n = 0
    # %%
    # raise RuntimeError("Stop here")
    losses, model = slicetca.decompose(
        trial_av,
        # neural_data_tensor,
        # all_con,
        # n_components,
        (n_components[0], 0, 0, 0),
        seed=best_seed,
        positive=True,
        # min_std=9e-3,
        # iter_std=20,
        learning_rate=1e-4,
        max_iter=100000,
        # batch_dim=3,
        # batch_prop=0.1,
        # batch_prop_decay=5,
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
                                eps=1e-9,
                                # weight_decay=1e-6
                             ),
        # weight_decay=partial(torch.optim.LBFGS, max_eval=200,
        #                      tolerance_grad=1e-6,
        #                      line_search_fn='strong_wolfe'),
        # mask=mask,
        # mask=all_mask,
        init_bias=0.1,
        initialization='uniform-positive',
        loss_function=torch.nn.L1Loss(reduction='mean'),#MovingAverageLoss(10),
        # loss_function=SoftDTW(True, 100, True, 20,
        #                       torch.nn.L1Loss(reduction='none')),#, _euclidean_squared_dist),
        # loss_function=partial(soft_dtw_normalized, gamma=1.0, normalize=True),
        verbose=0,
        compile=True,
        # shuffle_dim=(0, 1),
        device='cuda',
        # default_root_dir=os.path.join(os.path.dirname(LAB_root), 'logs'),
        gradient_clip_val=1,
        # accumulate_grad_batches=3,
        # reload_dataloaders_every_n_epochs=1,
        # regularization='L2',
        min_iter=5,
        # precision='16-mixed',
        dtype=torch.float32,
        testing=False,
    )
    # torch.save(model, f'model_{'SM'}_freq.pt')

    # plot the losses
    plt.figure(figsize=(4, 3), dpi=100)
    plt.plot(np.arange(10000, len(model.losses)), model.losses[10000:], 'k')
    plt.xlabel('iterations')
    plt.ylabel('mean squared error')
    plt.xlim(0, len(model.losses))
    plt.tight_layout()
    # # %% plot the model
    # idx1 = np.linspace(0, labels[0].shape[0], 8).astype(int)[1:-1]
    # idx2 = np.linspace(0, labels[1].shape[0], 6).astype(int)[1:-1]
    # timings = {'aud_ls': range(0, 200),
    #            'go_ls': range(200, 400),}
    #            # 'go_lm': range(800, 1000)}
    # components = model.get_components(numpy=True)
    # figs = {}
    # for cond, timing in timings.items():
    #     comp = model.get_components(numpy=True)
    #     comp[n] = [comp[n][1][..., timing]]
    #     # comp[n][1] = comp[n][2][..., timing]
    #     # comp[n][0] = np.array([])
    #     if cond.startswith('aud_l'):
    #         t_label = f"Time (s) from Stimulus"
    #     elif cond.startswith('go_lm'):
    #         t_label = f"Time (s) from Go Cue (Mime)"
    #     elif cond.startswith('go_ls'):
    #         t_label = f"Time (s) from Go Cue (Speak)"
    #     elif cond.startswith('go_jl'):
    #         t_label = f"Time (s) from Go Cue (:=:)"
    #     axes = slicetca.plot(model,
    #                          components=comp,
    #                          ignore_component=(0,),
    #                          variables=('channel',
    #                                     'freq',
    #                                     t_label),
    #                          sorting_indices=(None,
    #                                           None,
    #                                           # labels[1].astype(float).argsort()[::-1],
    #                                           None),
    #                          ticks=(None,
    #                                 None,
    #                                 # idx2[::-1],
    #                                 [0, 49, 99, 149, 199]),
    #                          tick_labels=(labels[0][idx1],
    #                                       None,
    #                                       # labels[1][idx2].astype(float).astype(int),
    #                                       [-0.5, 0, 0.5, 1, 1.5]),
    #                          cmap=parula_map)
    colors = ['orange', 'y', 'k', 'c', 'm', 'deeppink',
              'darkorange', 'lime', 'blue', 'red', 'purple']
    # # %% plot the sensory motors
    # plot_sm = False
    # if plot_sm:
    #     timingss = [{'aud_ls': range(0, 200),
    #                'go_ls': range(600, 800)},
    #                {'aud_jl': range(400, 600),
    #                 'go_jl': range(1000, 1200)}]
    #     fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    #     ylims = [0, 0]
    #     lss = ['-', '--']
    #     for i, timings in enumerate(timingss):
    #         for j, (cond, time_slice) in enumerate(timings.items()):
    #             ax = axs[j]
    #             mat = np.nanmean(zscores[cond][:,sorted(SM)].__array__(), axis=(0,2,3))
    #             plot_dist(mat,#(trial_av.mean(1)*2).detach().cpu().numpy()[..., time_slice],
    #                             ax=ax, color='red', linestyle=lss[i], label=cond[-2:],
    #                       times=(-0.5, 1.5))
    #             if cond.startswith('go'):
    #                 event = "Go Cue"
    #             elif cond.startswith('aud'):
    #                 event = "Stimulus"
    #             if i == 0:
    #                 if j == 0:
    #                     # ax.legend()
    #                     ax.set_ylabel("Z-Score (V)")
    #             elif i == 1:
    #                 if j == 0:
    #                     ax.legend(loc='best')
    #
    #                 ax.set_xlabel("Time(s) from " + event)
    #             ylim = ax.get_ylim()
    #             ylims[1] = max(ylim[1], ylims[1])
    #             ylims[0] = min(ylim[0], ylims[0])
    #     for ax in axs:
    #         ax.set_ylim(ylims)
    #         ax.axhline(0, color='k', linestyle='--')
    #     fig.suptitle(idx_name)

    # %% plot the components
    mode = 'weighted'
    plot_components(model, 'components', all_con[:, 0], True)
    plt.suptitle(f"Components")

    # %% plot the components
    W, H = model.get_components(numpy=True)[0]
    # W2 = (np.cov(
    #     model.forward().detach().cpu().numpy().reshape(W.shape[1],
    #                                                    -1)) @ W.T).T
    # H2 = H @ np.cov(
    #     model.forward().detach().cpu().numpy().reshape(-1, H.shape[-1]),
    #     rowvar=False)
    timingss = [{'aud_ls': slice(0, 200),
               'go_ls': slice(200, 400)},
                {'aud_lm': slice(0, 200),
                 'go_lm': slice(200, 400)},
                {'aud_jl': slice(0, 200),
                 'go_jl': slice(200, 400)}
                ]
    colors = colors[:n_components[0]]
    conds = {'aud_ls': (-0.5, 1.5),
             'go_ls': (-0.5, 1.5),
                'aud_lm': (-0.5, 1.5),
                'go_lm': (-0.5, 1.5),
             'aud_jl': (-0.5, 1.5),
             'go_jl': (-0.5, 1.5)}
    fig, axs = plt.subplots(n_components[n], 2, dpi=100)
    ylims = [[], []]
    # make a plot for each condition in conds as a subgrid
    for i, timings in enumerate(timingss):
        for j, (cond, times) in enumerate(timings.items()):
            for k in range(n_components[n]):
                ax = axs[k, j]
                if i == 0:
                    ls = '-'
                elif i == 1:
                    ls = '--'
                else:
                    ls = ':'
                plot_dist(
                    to_plot(mode, W, H, all_con[:, i], timings, k, cond, i),
                    ax=ax, color=colors[k], mode='sem', times=(-0.5, 1.5),
                    linestyle=ls, label=cond[3:])
                ylim = ax.get_ylim()
                ylims[1].append(ylim[1])
                ylims[0].append(ylim[0])
                if j == 1 and i == 2:
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
    W, H = model.get_components(numpy=True)[0]
    W2 = (np.cov(
        model.forward().detach().cpu().numpy().reshape(W.shape[1],
                                                       -1)) @ W.T).T
    chans = ['-'.join([f"D{int(ch.split('-')[0][1:])}", ch.split('-')[1]]) for
             ch in labels[0]]
    subj = sorted(set(ch.split('-')[0] for ch in chans))
    electrode_gradient(layout.get_subjects(), W * 2, chans, colors, mode='both', fig_dims=(W.shape[0],1))

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
    sm_idx = [labels[0].tolist().index(l) for l in lzfilled]
    plot_on_average(layout.get_subjects(),
                    picks=[f"D{int(xi[1:5])}{xi[5:]}" for xi in
                           labels[0][torch.tensor(sm_idx)[
                               (W[1] / W.sum(0)) > 0.45]]], hemi='both',
                    size=W[0, (W[1] / W.sum(0)) > 0.45], label_every=1)
    idxs = [torch.tensor(sm_idx)[
                (W[i] / W.sum(0)) > 0.4
                # W.argmax(0) == i
                ].tolist() for i in range(n_components[0])]
    ylims = [0, 0]
    all_groups = []
    for idx in idxs:
        sm_elecs = labels[0][idx]
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

    # %% plot each component trials
    W, H = model.get_components(numpy=True)[0]
    colors = ['orange', 'y', 'k', 'c', 'm', 'deeppink',
              'darkorange', 'lime', 'blue', 'red', 'purple'
                ]
    colors = colors[:W.shape[0]]
    timings = {'aud_ls': range(0, 200),
                'go_ls': range(200, 400)}
    conds = {'aud_ls': (-0.5, 1.5),
             'go_ls': (-0.5, 1.5)}
    n_comp = W.shape[0]
    fig, axs = plt.subplots(2*len(timings), n_comp)
    data = neural_data_tensor.nanmean((1, 2)).detach().cpu().numpy()
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

            scale = np.mean(sorted_trimmed) + 1.5 * np.std(sorted_trimmed)
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
