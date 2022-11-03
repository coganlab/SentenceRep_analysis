import matplotlib.pyplot as plt
from utils.calc import get_elbow, dist, do_decomp, par_calc, ArrayLike
from sklearn.decomposition import NMF
from utils.mat_load import group_elecs, get_sigs, load_all
import numpy as np
from typing import Union, Iterable


def plot_decomp(data: ArrayLike, clusters: int = 8, repetitions: int = 10,
                mod=NMF(init='random', max_iter=10000, verbose=2)):
    """Plots optimal K based on explained varience"""
    errs = do_decomp(data, clusters, repetitions, mod)
    plot_dist(errs)
    plt.xlabel("K Value")
    plt.xticks(np.array(range(clusters)), np.array(range(clusters)) + 1)
    plt.show()


def plot_dist(mat: iter, mask: ArrayLike = None, label: Union[str, int, float] = None,
              color: Union[str, list[int]] = None) -> plt.Axes:
    """Plot the distribution for a single signal"""
    mean, std = dist(mat, mask)
    tscale = range(len(mean))
    plt.errorbar(tscale, mean, yerr=std, label=label, color=color)
    return plt.gca()


def plot_factors(factors: list[ArrayLike],
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


def plot_weight_dist(data: ArrayLike, label: ArrayLike, mask: ArrayLike = None,
                     sig_titles: list[str] = None, colors: list[Union[str, list[int]]] = None):
    """Basic distribution plot for weighted signals"""
    fig, ax = plt.subplots()
    if label.shape[0] > 1:
        group = range(min(np.shape(label)))
        weighted = True
    else:
        group = np.unique(label)
        weighted = False
    if sig_titles is None:
        sig_titles = [sig_titles] * len(group)
    if colors is None:
        colors = [colors] * len(group)
    for i, stitle, color in zip(group, sig_titles, colors):
        if not weighted:
            w_sigs = data[label == i]
        else:
            w_sigs = np.multiply(data.T, label[:, i]).T
            # try:
            #     w_sigs = np.array([label[i][j] * dat for j, dat in enumerate(data.T)])
            # except (ValueError, IndexError) as e:
            #     w_sigs = np.array([label.T[i][j] * dat for j, dat in enumerate(data)])
        ax = plot_dist(w_sigs, mask, stitle, color)
    return fig, ax


def plot_clustering(data: ArrayLike, label: ArrayLike, mask: ArrayLike = None,
                    sig_titles: Iterable[str] = None,
                    colors: Iterable[Union[str, list[Union[int, float]]]] = None):
    """Stylized multiplot for clustering"""
    fig, ax = plot_weight_dist(data, label, mask, sig_titles, colors)
    # the x coords of this transformation are data, and the
    # y coord are axes
    trans = ax.get_xaxis_transform()
    ax.text(50, 0.8, 'Stim onset', rotation=270, transform=trans)
    ax.axvline(175)
    ax.axvline(50, linestyle='--')
    ax.axvline(225, linestyle='--')
    # ax.axhline(0, linestyle='--', color='black')
    ax.text(225, 0.87, 'Go cue', rotation=270, transform=trans)
    ax.text(160, 0.6, 'Delay', transform=trans)
    # ax.legend(loc="best")
    ax.axvspan(150, 200, color=(0.5, 0.5, 0.5, 0.15))
    ax.set_xticks([0, 50, 100, 150, 200, 225, 250, 300, 350],
                  ['-0.5', '0', '0.5', '1', '-0.25', '0', '0.25', '0.75', '1.25'])
    ax.set_xlabel('Time from stimuli or go cue (seconds)')
    # ax.set_ylabel('Z score')
    ax.set_ylabel('Z-score')
    ax.set_xlim(0, 350)
    ylims = ax.get_ybound()
    ax.set_ybound(min(0, ylims[0]), ylims[1])
    # plt.title(title)
    plt.show()
    return fig, ax


def plot_clustering_resp(data: ArrayLike, label: ArrayLike,
                         sig_titles: Iterable[str] = None, weighted: bool = False,
                         colors: Iterable[Union[str, list[Union[int, float]]]] = None, ybounds=None):
    fig, ax = plot_weight_dist(data, label, sig_titles, colors, weighted)
    trans = ax.get_xaxis_transform()
    ax.text(100, 0.9, 'onset', rotation=270, transform=trans)
    ax.axvline(100, linestyle='--')
    # ax.axvline(50, linestyle='--')
    # ax.axvline(225, linestyle='--')
    # ax.text(225, 0.87, 'go cue', rotation=270, transform=trans)
    # ax.text(152, 0.6, 'transition',  transform=trans)
    ax.legend(loc="best")
    # ax.axvspan(150,200,color=(0.5,0.5,0.5,0.15))
    ax.set_ybound(ybounds)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100, 150, 200],
                  ['-1', '-0.5', '0', '0.5', '1'])
    ax.set_xlabel('Time from response (seconds)')
    # ax.set_ylabel('Significance of activity (range [0-1])')
    ax.set_xlim(0, 200)
    # if title is not None:
    #     plt.title(title)
    plt.show()


def plot_opt_k(data: ArrayLike, n: int, rep: int, model, methods=None, title=None):
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


def alt_plot(X_train: ArrayLike, y_pred: ArrayLike):
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
    Task, all_sigZ, all_sigA, sig_chans, sigMatChansLoc, sigMatChansName, Subject = load_all('data/pydata.mat')
    SM, AUD, PROD = group_elecs(all_sigA, sig_chans)
    npSM = np.array(SM)
    cond = 'LSwords'
    SMresp = npSM[npSM < len(all_sigA[cond]['Response'])]
    resp = all_sigA[cond]['Response'][SMresp, :]
    sigZ, sigA = get_sigs(all_sigZ, all_sigA, sig_chans, cond)
    winners, results, w_sav = np.load('data/nmf.npy', allow_pickle=True)
    SMrespw = w_sav['SM'][npSM < len(all_sigA['LSwords']['Response'])]
    ones = np.ones([244, 1])
    x = np.array(sigZ['AUD'])
    labels = np.ones([np.shape(sigZ['AUD'])[0]])
    x = np.vstack([x, np.array(sigZ['SM'])])
    labels = np.concatenate([labels, np.ones([np.shape(sigZ['SM'])[0]]) * 2])
    x = np.vstack([x, np.array(sigZ['PROD'])])
    labels = np.concatenate([labels, np.ones([np.shape(sigZ['PROD'])[0]]) * 3])
    colors = [[0, 0, 0], [0.6, 0.3, 0], [.9, .9, 0], [1, 0.5, 0]]
    names = ['Working Memory','Visual','Motor','Auditory']
    # plot_clustering(sigA['SM'], np.ones([244, 1]), None, True, [[1, 0, 0]])
    plot_clustering(sigZ['SM'], w_sav['SM'], sigA['SM'], sig_titles=names, colors=colors)
    plt.legend(loc="best")
    plot_weight_dist(all_sigZ[cond]['Response'][SMresp,:], SMrespw, resp, sig_titles=names, colors=colors)
    # [[0,1,0],[1,0,0],[0,0,1]])
    ax = plt.gca()
    ylims = list(ax.get_ybound())
    ylims[0] = min(0, ylims[0])
    # plt.title('Listen-speak')
    # plot_clustering_resp(resp, np.ones([180, 1]), ['SM'], True,
    #                      [[1, 0, 0]], ylims)