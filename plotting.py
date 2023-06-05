
from utils.calc import do_decomp, par_calc
from sklearn.decomposition import NMF
import numpy as np
from ieeg.calc.stats import dist
from ieeg.calc.utils import get_elbow
from ieeg.viz.utils import plot_dist, plot_weight_dist
import matplotlib.pyplot as plt
from collections.abc import Iterable


def plot_decomp(data: np.ndarray, clusters: int = 8, repetitions: int = 10,
                mod=NMF(init='random', max_iter=10000, verbose=2)):
    """Plots optimal K based on explained varience"""
    errs = do_decomp(data, clusters, repetitions, mod)
    plot_dist(errs)
    plt.xlabel("K Value")
    plt.xticks(np.array(range(clusters)), np.array(range(clusters)) + 1)
    plt.show()


def plot_factors(factors: list[np.ndarray],
                 col_titles: list[str] = ("Channel", "Trial", "Time")):
    fig, axs = plt.subplots(factors[0].shape[1], len(factors))
    for i, axr in enumerate(axs):
        for j, axc in enumerate(axr):
            axc.plot(factors[j][:,i])
            # if j == 0:
            #     plt.ylabel("factor " + str(i))
            # if i == 0:
            #     plt.title(col_titles[j])
    return fig, axs


def compare_subjects(data: np.ndarray, names: list[str], subj_per_plot: int = 8):
    """Plots the average signal for each subject in data against the average of all subjects

    Parameters
    ----------
    data : np.ndarray
        A numpy array of all the significant channels to plot (n_channels, n_timepoints)
    names : str
        The channel names for each channel in data, including the subject name (n_channels)
    subj_per_plot : int, optional
        The number of subjects to plot in each figure, by default 5
        """

    sub_all = list({name.split('-')[0] for name in names})
    sub_all.sort()

    fig, axs = plt.subplots(1, int(np.ceil(len(sub_all) / subj_per_plot)),
                            sharex=True, sharey=True)
    if not isinstance(axs, Iterable):
        axs = [axs]

    subj_data = np.zeros((0, data.shape[1]))
    prev_sub = names[0].split("-")[0]
    for c, name in zip(range(data.shape[0]), names):
        subj = name.split('-')[0]
        chan = data[c]
        plt_idx = int(sub_all.index(subj) / subj_per_plot)

        # new subject
        if subj != prev_sub or name == names[-1]:
            # new subplot
            if sub_all.index(prev_sub) % subj_per_plot == 0:
                fig.sca(axs[plt_idx])
                plot_dist(data, label='avg')
            plot_dist(subj_data, label=prev_sub)
            subj_data = np.zeros((0, data.shape[1]))

        subj_data = np.vstack([subj_data, chan])
        prev_sub = subj
    for ax in axs:
        ax.legend()


def plot_clustering(data: np.ndarray, label: np.ndarray, mask: np.ndarray = None,
                    sig_titles: list[str] = None,
                    colors: list[str | list[int | float]] = None):
    """Stylized multiplot for clustering"""
    fig, ax = plot_weight_dist(data, label, mask, sig_titles, colors)
    # the x coords of this transformation are data, and the
    # y coord are axes
    trans = ax.get_xaxis_transform()
    ax.text(50, 0.8, 'Stim onset', rotation=270, transform=trans)
    ax.axvline(175)
    ax.axvline(50, linestyle='--')
    ax.axvline(200, linestyle='--')
    # ax.axhline(0, linestyle='--', color='black')
    ax.text(200, 0.87, 'Go cue', rotation=270, transform=trans)
    ax.text(160, 0.6, 'Delay', transform=trans)
    # ax.legend(loc="best")
    ax.axvspan(150, 200, color=(0.5, 0.5, 0.5, 0.15))
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350],
                  ['-0.5', '0', '0.5', '1', '0', '0.5', '1', '1.5'])
    ax.set_xlabel('Time from stimuli or go cue (seconds)')
    # ax.set_ylabel('Z score')
    ax.set_ylabel('Z-score')
    ax.set_xlim(0, 350)
    ylims = ax.get_ybound()
    ax.set_ybound(min(0, ylims[0]), ylims[1])
    # plt.title(title)
    plt.show()
    return fig, ax


def plot_opt_k(data: np.ndarray, n: int, rep: int, model, methods=None, title=None):
    if methods is None:
        methods = ['euclidean', 'dtw', 'softdtw']
    if title is None:
        title = str(len(data))
    title = "Optimal k for " + title
    results = {}
    for method in methods:
        model.metric = method
        sil, var, wss = par_calc(data, n, rep, model, method)
        score = {'sil': sil, 'var': var, 'wss': wss}
        for key, value in score.items():
            mean, std = dist(value)
            tscale = range(len(mean))
            plt.errorbar(tscale, mean, yerr=std)
            plt.ylabel(method + key)
            plt.xlabel("K Value")
            plt.xticks(np.array(range(n)), np.array(range(n)) + 1)
            plt.title(title)
            plt.show()
        score['k'] = get_elbow(np.mean(sil, 0)) + 1
        results[method] = score
    return results


def alt_plot(X_train: np.ndarray, y_pred: np.ndarray):
    plt.figure()
    for yi in range(len(np.unique(y_pred))):
        plt.subplot(len(np.unique(y_pred)), 1, 1 + yi)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.xlim(0, X_train.shape[1])
        plt.title("Cluster %d" % (yi + 1))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import matplotlib as mpl
    mpl.use('TkAgg', force=True)
    import mne
    import os
    import numpy as np
    from ieeg.io import get_data
    from ieeg.viz.utils import plot_dist
    from ieeg.viz.mri import get_sub_dir, plot_on_average, gen_labels
    import matplotlib.pyplot as plt
    from utils.mat_load import load_intermediates, group_elecs

    # %% check if currently running a slurm job
    HOME = os.path.expanduser("~")
    if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
        LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    else:  # if not then set box directory
        LAB_root = os.path.join(HOME, "Box", "CoganLab")
    layout = get_data("SentenceRep", root=LAB_root)
    conds = {"resp": (-1, 1),
             "aud_ls": (-0.5, 1.5),
             "aud_lm": (-0.5, 1.5),
             "aud_jl": (-0.5, 1.5),
             "go_ls": (-0.5, 1.5),
             "go_lm": (-0.5, 1.5),
             "go_jl": (-0.5, 1.5)}

    # %% Load the data
    # epochs, all_power, names = load_intermediates(layout, conds, "zscore")
    signif, all_sig, names = load_intermediates(layout, conds, "significance")
    AUD, SM, PROD, sig_chans = group_elecs(all_sig, names, conds)

    # %%
    data = [v['resp'] for v in signif.values() if v]
    plot_on_average(data, picks=PROD)
