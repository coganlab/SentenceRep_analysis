from analysis.utils.calc import do_decomp, par_calc
from sklearn.decomposition import NMF
import numpy as np
from ieeg.calc.stats import dist
from ieeg.calc.mat import get_elbow, LabeledArray
from ieeg.viz.ensemble import plot_dist, plot_weight_dist
import matplotlib.pyplot as plt
from collections.abc import Iterable
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


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
        subj = name.split('-', )[0]
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

def boxplot_2d(x,y, ax, whis=1.5):
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    #the box
    box = Rectangle(
        (xlimits[0],ylimits[0]),
        (xlimits[2]-xlimits[0]),
        (ylimits[2]-ylimits[0]),
        ec = 'k',
        zorder=0
    )
    ax.add_patch(box)

    #the x median
    vline = Line2D(
        [xlimits[1],xlimits[1]],[ylimits[0],ylimits[2]],
        color='k',
        zorder=1
    )
    ax.add_line(vline)

    #the y median
    hline = Line2D(
        [xlimits[0],xlimits[2]],[ylimits[1],ylimits[1]],
        color='k',
        zorder=1
    )
    ax.add_line(hline)

    #the central point
    ax.plot([xlimits[1]],[ylimits[1]], color='k', marker='o')

    #the x-whisker
    #defined as in matplotlib boxplot:
    #As a float, determines the reach of the whiskers to the beyond the
    #first and third quartiles. In other words, where IQR is the
    #interquartile range (Q3-Q1), the upper whisker will extend to
    #last datum less than Q3 + whis*IQR). Similarly, the lower whisker
    #will extend to the first datum greater than Q1 - whis*IQR. Beyond
    #the whiskers, data are considered outliers and are plotted as
    #individual points. Set this to an unreasonably high value to force
    #the whiskers to show the min and max values. Alternatively, set this
    #to an ascending sequence of percentile (e.g., [5, 95]) to set the
    #whiskers at specific percentiles of the data. Finally, whis can
    #be the string 'range' to force the whiskers to the min and max of
    #the data.
    iqr = xlimits[2]-xlimits[0]

    #left
    left = np.min(x[x > xlimits[0]-whis*iqr])
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1],ylimits[1]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [left, left], [ylimits[0],ylimits[2]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_bar)

    #right
    right = np.max(x[x < xlimits[2]+whis*iqr])
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1],ylimits[1]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [right, right], [ylimits[0],ylimits[2]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_bar)

    #the y-whisker
    iqr = ylimits[2]-ylimits[0]

    #bottom
    bottom = np.min(y[y > ylimits[0]-whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [bottom, ylimits[0]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [bottom, bottom],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##top
    top = np.max(y[y < ylimits[2]+whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [top, ylimits[2]],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [top, top],
        color = 'k',
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##outliers
    mask = (x<left)|(x>right)|(y<bottom)|(y>top)
    ax.scatter(
        x[mask],y[mask],
        facecolors='none', edgecolors='k'
    )


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


def plot_channels(arr: LabeledArray, sig: LabeledArray = None,
                  n_cols: int = 10, n_rows: int = 6,
                  size: tuple[int, int] = (8, 12), **kwargs):
    # n_rows = int(np.ceil(arr.shape[-3] / n_cols))
    per_fig = n_cols * n_rows
    numfigs = int(np.ceil(arr.shape[-3] / per_fig))
    figs = []
    for i in range(numfigs):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=size, frameon=False,
                                layout='tight')
        for j, ax in enumerate(axs.flatten()):
            sig_num = j + i * per_fig
            if sig_num >= arr.shape[-3]:
                break
            plot_dist(arr[sig_num].__array__(), axis=0, ax=ax, mode='std', **kwargs)
            if sig is not None:
                plot_horizontal_bars(ax, [sig[sig_num].__array__()], kwargs['times'])
            ax.set_title(arr.labels[-3][sig_num])
        figs.append(fig)
    return figs


def plot_horizontal_bars(ax: plt.Axes,
                         where: list[np.ndarray[bool], ...], times: list[float] = None,
                         bar_height=None, location='below'):
    """Plot horizontal bars on an axis according to a boolean array

    Parameters
    ----------
    ax : plt.Axes
        The axis to plot on
    where : list[np.ndarray[bool], ...]
        A list of boolean arrays, where each array is the same length as the
        number of bars on the axis. The bars will be plotted where the array
        is True.
    bar_height : float, optional
        The height of the bars, by default 0.2
    location : str, optional
        Where to plot the bars, by default 'below'
    """

    lines = ax.get_lines()
    for i, line in enumerate(lines):
        x = line.get_xdata()
        width = x[1] - x[0]
        color = line.get_color()
        ylims = ax.get_ylim()
        if bar_height is None:
            bar_height = (ylims[1] - ylims[0]) / 12
        if location == 'below':
            y0 = ylims[0] + (i + 1) * bar_height
            y0 -= bar_height / 2
        else:
            y0 = ylims[1] - (i + 1) * bar_height
            y0 += bar_height / 2
        for j in range(len(x)):
            if where[i][j]:
                ax.barh(y0, width=width, left=x[j], height=bar_height,
                        color=color)


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
    from analysis.utils.mat_load import load_intermediates, group_elecs

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
    no_plot = [f'sub-{id}' for id in ['D0003', 'D0072']]
    data = [v['resp'] for v in signif.values() if v]
    plot_data = [v for v in data if v.info['subject_info']['his_id'] not in no_plot]
    brain = plot_on_average(data, picks=SM, color='red')
    plot_on_average(data, picks=AUD, color='green', fig=brain)
    plot_on_average(data, picks=PROD, color='blue', fig=brain)
