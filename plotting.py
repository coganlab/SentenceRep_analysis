
from utils.calc import do_decomp, par_calc
from sklearn.decomposition import NMF
from utils.mat_load import group_elecs
import numpy as np
from ieeg.calc.stats import dist
from ieeg.calc.utils import get_elbow


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


# def plot_clustering_resp(data: np.ndarray, label: np.ndarray,
#                          sig_titles: Iterable[str] = None, weighted: bool = False,
#                          colors: Iterable[Union[str, list[Union[int, float]]]] = None, ybounds=None):
#     fig, ax = plot_weight_dist(data, label, sig_titles, colors, weighted)
#     trans = ax.get_xaxis_transform()
#     ax.text(100, 0.9, 'onset', rotation=270, transform=trans)
#     ax.axvline(100, linestyle='--')
#     # ax.axvline(50, linestyle='--')
#     # ax.axvline(225, linestyle='--')
#     # ax.text(225, 0.87, 'go cue', rotation=270, transform=trans)
#     # ax.text(152, 0.6, 'transition',  transform=trans)
#     ax.legend(loc="best")
#     # ax.axvspan(150,200,color=(0.5,0.5,0.5,0.15))
#     ax.set_ybound(ybounds)
#     ax.set_yticks([])
#     ax.set_xticks([0, 50, 100, 150, 200],
#                   ['-1', '-0.5', '0', '0.5', '1'])
#     ax.set_xlabel('Time from response (seconds)')
#     # ax.set_ylabel('Significance of activity (range [0-1])')
#     ax.set_xlim(0, 200)
#     # if title is not None:
#     #     plt.title(title)
#     plt.show()


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
    from ieeg.viz.utils import plot_dist, plot_clustering
    from ieeg.viz.mri import get_sub_dir, plot_on_average
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
    epochs, all_power, names = load_intermediates(layout, conds, "zscore")

    data = {"D" + str(int(k[1:])): v['resp'] for k, v in epochs.items() if v}
    # %%
    plot_on_average(data, get_sub_dir())