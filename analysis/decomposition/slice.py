import torch
from analysis.grouping import GroupData
import os
import numpy as np
import matplotlib.pyplot as plt
import slicetca
from multiprocessing import freeze_support
from ieeg.viz.ensemble import plot_dist
from analysis.data import dataloader
from slicetca.invariance.iterative_invariance import within_invariance
from lightning.pytorch import Trainer
from analysis.decoding.models import SimpleDecoder
from analysis.decoding.train import process_data
from ieeg.calc.mat import LabeledArray
from analysis.decoding import windower
from joblib import Parallel, delayed
import logging

import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['C:\\Users\\Jakda\\git'])

# ## set up the figure
fpath = os.path.expanduser("~/Box/CoganLab")
# # Create a gridspec instance with 3 rows and 3 columns
device = ('cuda' if torch.cuda.is_available() else 'cpu')
#


# %% Load the data

if __name__ == '__main__':
    freeze_support()
    sub = GroupData.from_intermediates("SentenceRep", fpath, folder='stats')
    sm_idx = sorted(list(sub.SM))
    aud_slice = slice(0, 175)
    conds = ['aud_ls', 'aud_lm', 'go_ls', 'go_lm', 'resp']
    neural_data_tensor, labels = dataloader(sub, sm_idx, conds)
    mask = ~torch.isnan(neural_data_tensor)
    neural_data_tensor[torch.isnan(neural_data_tensor)] = 0

    n_components = (5,)
    best_seed = 406579

    # #
    # %% decompose the optimal model

    losses, model = slicetca.decompose(neural_data_tensor,
                                       n_components,
                                       # (0, n_components[0], 0),
                                       seed=best_seed,
                                       positive=True,
                                       # min_std=5e-5,
                                       # iter_std=1000,
                                       learning_rate=1e-2,
                                       max_iter=10000,
                                       # batch_dim=0,
                                       batch_prop=0.33,
                                       batch_prop_decay=3,
                                       # weight_decay=1e-3,
                                       mask=mask,
                                       init_bias=0.01,
                                       initialization='uniform-positive',
                                       loss_function=torch.nn.HuberLoss(reduction='mean'),
                                       verbose=0
                                       )
    # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    # slicetca.invariance(model, L3 = None)
    # %% plot the losses
    plt.figure(figsize=(4, 3), dpi=100)
    plt.plot(np.arange(50, len(model.losses)), model.losses[50:], 'k')
    plt.xlabel('iterations')
    plt.ylabel('mean squared error')
    plt.xlim(0, len(model.losses))
    plt.tight_layout()
    # %% plot the model
    axes = slicetca.plot(model,
                         variables=('trial', 'neuron', 'time'),)
    colors = ['orange', 'y', 'k', 'c', 'm']
    T, W, H = model.get_components(numpy=True)[0]
    # %% plot the components
    colors = colors[:n_components[0]]
    conds = {'aud_ls': (-0.5, 1.5),
             'go_ls': (-0.5, 1.5),
             'resp': (-1, 1)}
    timings = {'aud_ls': range(0, 200),
               'go_ls': range(400, 600),
               'resp': range(800, 1000)}
    fig, axs = plt.subplots(1, 3)

    # make a plot for each condition in conds as a subgrid
    for j, (cond, times) in enumerate(conds.items()):
        ax = axs[j]
        for i in range(n_components[0]):
            fig = plot_dist(
                # H[i],
                model.construct_single_component(0, i).detach().cpu().numpy()[:, (W[i] / W.sum(0)) > 0.4][
                    ..., timings[cond]].reshape(-1, 200),
                ax=ax, color=colors[i], mode='std', times=times, label=f"Component {colors[i]}")
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
             ch in sub.array.labels[3][sm_idx]]
    electrode_gradient(sub.subjects, W, chans, colors, mode='both')

    # %% plot each component
    n = W.shape[0]
    for cond, times in timings.items():
        fig, axs = plt.subplots(2, n)

        data = neural_data_tensor.mean(0).detach().cpu().numpy()
        ylims = [0, 0]
        for i, ax in enumerate(axs[0]):
            component = model.construct_single_component(0, i).detach().cpu().numpy()
            trimmed = data[(W[i] / W.sum(0)) > 0.4][:, times]
            sorted_trimmed = trimmed[np.argsort(W[i, (W[i] / W.sum(0)) > 0.4])][::-1]
            plot_dist(trimmed.reshape(-1, 200), ax=ax, color=colors[i], mode='std', times=conds[cond])
            ax.set_xticks([])
            ylims[1] = max(ax.get_ylim()[1], ylims[1])
            ylims[0] = min(ax.get_ylim()[0], ylims[0])

            scale = np.mean(sorted_trimmed) + 2 * np.std(sorted_trimmed)
            axs[1, i].imshow(sorted_trimmed, aspect='auto', cmap='inferno', vmin=0, vmax=scale,
                             extent=[conds[cond][0], conds[cond][-1], 0, len(sorted_trimmed)])

            if i > 0:
                axs[0, i].set_yticks([])
                axs[1, i].set_yticks([])

        for ax in axs[0]:
            ax.set_ylim(ylims)
        fig.suptitle(cond)
        fig.tight_layout()

    # %% plot the region membership
    from ieeg.viz.mri import gen_labels, subject_to_info, BN_2_roi

    colors = ['WM', 'Late Prod', 'Instructional', 'Feedback', 'Early Prod']
    rois = ['mtg', 'astg', 'pstg', 'heschl', 'sts', 'itg', 'ipc', 'angular', 'supramarginal', 'ifg', 'opercular',
            'triangular', 'ifs', 'ifj', 'mfg', 'insula', 'sma', 'smc', 'spl', 'pcl', 'occ', 'hipp', 'bg', 'sfg']
    groups = {r: [] for r in rois}
    fig, axs = plt.subplots(1, 5)
    idxs = [torch.tensor(sm_idx)[(W[i] / W.sum(0)) > 0.5].tolist() for i in range(5)]
    ylims = [0, 0]
    for c, ax, idx in zip(colors, axs, idxs):
        sm_elecs = sub.array[:, :, :, idx].labels[3]
        for subj in sub.subjects:
            subj_old = f"D{int(subj[1:])}"
            info = subject_to_info(subj_old)
            ch_labels = gen_labels(info, subj_old, atlas='.BN_atlas')
            for key, value in ch_labels.items():
                item = subj + '-' + key
                if item in sm_elecs:
                    area = BN_2_roi(value.split("_")[0])
                    if area in groups.keys():
                        groups[area].append(subj + '-' + key)
        groups_num = {key: len(value) for key, value in groups.items()}
        # plot the histogram, with
        ax.bar(groups_num.keys(), groups_num.values())
        # make x axis labels sideways
        plt.setp(ax.get_xticklabels(), rotation=90, ha='left')
        ax.set_title(f"Component {c}")
        ylim = ax.get_ylim()
        ylims[1] = max(ylim[1], ylims[1])
        ylims[0] = min(ylim[0], ylims[0])
    for ax in axs:
        ax.set_ylim(ylims)
    plt.tight_layout()

    # %% save the model
    torch.save(model, 'model.pt')

    # # %% varimax rotation
    # # create a copy of the model
    # model_rot = model.copy()
    # # rotate the components
    # target_map = {'heat': 0, 'hut': 1, 'hot': 2, 'hoot': 3}
    # target_labels = (x.split('-')[0] for x in labels[0])
    # targets = torch.tensor([target_map[x] for x in target_labels])
    # # targets = torch.tensor([[target_map[x] for x in target_labels]] * 800)
    # decoders = [SimpleDecoder(4, len(labels[2]), 1e-4) for _ in range(n_components[0])]
    # logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    # logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
    # def loss(x: list[torch.Tensor]):
    #     results = []
    #     # decoders = [SimpleDecoder(4, len(labels[2]), 5e-4) for _ in range(n_components[0])]
    #     for decoder, xi in zip(decoders, x):
    #         # decoder = SimpleDecoder(4, xi.shape[1] * xi.shape[2], 5e-3)
    #         decoder = decoders[i]
    #         temp = xi.clone().detach()
    #         # train the decoder briefly
    #         Trainer(max_epochs=1000,
    #                 devices=1,
    #                 accelerator=device,
    #                 barebones=True,
    #                 # precision=32,
    #                 limit_train_batches=1).fit(decoder, (temp, targets))
    #         y_hat = decoder(xi)
    #         results.append(decoder.criterion(y_hat, targets))
    #
    #     results.sort()
    #     print([r.item() for r in results])
    #     return results[0]
    # model_rot = within_invariance(model_rot, loss, maximize=False, max_iter=100, ignore=(1,))
    # T, W, H = model_rot.get_components(numpy=True)[0]
    #
    # # %% plot the components
    # colors = colors[:n_components[0]]
    # conds = {'aud_ls': (-0.5, 1.5),
    #          'go_ls': (-0.5, 1.5),
    #          'resp': (-1, 1)}
    # fig, axs = plt.subplots(1, 4)
    #
    # # make a plot for each condition in conds as a subgrid
    # for j, (cond, times) in enumerate(conds.items()):
    #     ax = axs[j]
    #     start = 200 * j
    #     end = start + 200
    #     for i in range(n_components[0]):
    #         fig = plot_dist(
    #             # H[i],
    #             model.construct_single_component(0, i).detach().numpy()[:,
    #             (W[i] / W.sum(0)) > 0.3, start:end].reshape(-1, 200),
    #             ax=ax, color=colors[i], mode='std', times=times)
    #     if j == 0:
    #         ax.legend()
    #         ax.set_ylabel("Z-Score (V)")
    #         ylims = ax.get_ylim()
    #     elif j == 1:
    #         ax.set_xlabel("Time(s)")
    #     ax.set_ylim(ylims)
    #     ax.set_title(cond)

    # %% windowed decoding aud

    n_folds = 5
    val_size = 1/n_folds
    max_epochs = 500
    results = {str(i): None for i in range(n_components[0])}
    target_map = {'heat': 0, 'hut': 1, 'hot': 2, 'hoot': 3}
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
    decoders = [SimpleDecoder(4, len(labels[2]), 5e-4) for _ in range(n_components[0])]
    for i, decoder in enumerate(decoders):
        # xi = model.construct_single_component(0, i).detach().cpu().numpy().swapaxes(0, 1)
        xi = neural_data_tensor.clone().detach().cpu().numpy().swapaxes(0, 1)
        trimmed = xi[(W[i] / W.sum(0)) > 0.4]
        sorted_trimmed = trimmed[np.argsort(W[i, (W[i] / W.sum(0)) > 0.4])][
                         ::-1]
        ls = sorted_trimmed[..., :200]
        lm = sorted_trimmed[..., 200:400]
        stacked = np.concatenate([ls, lm], axis=1)
        data_windowed = LabeledArray(windower(stacked, 20, 2).swapaxes(0, -1))[::5]
        data_windowed.labels[2] = np.concatenate([labels[0], labels[0]])
        data_windowed.labels[1] = labels[1]
        # decoder = SimpleDecoder(4, xi.shape[1] * xi.shape[2], 5e-3)
        # train the decoder briefly
        out = Parallel(n_jobs=-2, verbose=40)(delayed(process_data)(
            d, 5, n_folds, val_size, target_map, max_epochs) for d in
                                             data_windowed)
        results[str(i)] = [o.tolist() for o in out]

    # %% plot the results
    fig, ax = plt.subplots(1, 1)
    for i, res in results.items():
        plot = torch.tensor(res).flatten(1).T
        fig = plot_dist(plot.detach().cpu().numpy(), times=(-0.4, 1.4), ax=ax)
        fig.title.set_text("Decoding accuracy")

    # import pickle
    # with open('results2.pkl', 'wb') as f:
    #     pickle.dump(results, f)
