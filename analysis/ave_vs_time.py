from analysis.grouping import GroupData
from analysis.utils.plotting import plot_channels
import os
from analysis.decoding.words import score
from os.path import join
# %% pick strategy

HOME = os.path.expanduser("~")

if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    box = join(HOME, "workspace")
    ids = [int(os.environ['SLURM_ARRAY_TASK_ID'])]

else:  # if not then set box directory
    box = join(HOME, "Box")
    ids = list(range(3))

# %% Imports

fpath = join(box, "CoganLab")
subjects_dir = join(box, "ECoG_Recon")
for id in ids:
    if id == 0:
        sub = GroupData.from_intermediates("SentenceRep", fpath,
                                           folder='stats_opt',
                                           subjects_dir=subjects_dir)
        name = "wide"
    elif id == 1:
        sub = GroupData.from_intermediates("SentenceRep", fpath, fdr=True,
                                           folder='no_cluster',
                                           subjects_dir=subjects_dir)
        name = "ave"
    elif id == 2:
        sub = GroupData.from_intermediates("SentenceRep", fpath, wide=True,
                                           folder='stats_opt',
                                           subjects_dir=subjects_dir)
        name = "short"
    else:
        raise ValueError(f"ID {id} not recognized")

    # fig = sub.plot_groups_on_average(rm_wm=False)

    # %% Imports
    all_data = []
    colors = [[0, 1, 0], [1, 0, 0], [0, 0, 1], [0.5, 0.5, 0.5]]
    scores = {'Auditory': None, 'Sensory-Motor': None, 'Production': None, 'All': None}
    idxs = [sub.AUD, sub.SM, sub.PROD, sub.sig_chans]
    idxs = [list(idx & sub.grey_matter) for idx in idxs]
    names = list(scores.keys())
    conds = [['aud_ls', 'aud_lm'], ['go_ls', 'go_lm'], 'resp']
    window_kwargs = {'window': 20, 'obs_axs': 1, 'normalize': 'true', 'n_jobs': -3,
                     'average_repetitions': False}
    cats = {'heat': 1, 'hoot': 2, 'hot': 3, 'hut': 4}

    # %% run the modeling
    fname = join(HOME, f"true_scores_{name}.npy")
    score(cats, 0.8, 'lda', 5, 10, sub, idxs,
          conds, window_kwargs, fname, shuffle=False)
    # fname = join(HOME, f"shuffle_score_{name}.npy")
    # score(cats, 0.8, 'lda', 5, 250, sub,
    #       idxs, conds, window_kwargs, fname, shuffle=True)

    # # %% Plot all channels
    # x = sub.array["zscore", "aud_ls", :, sub.AUD]
    # x.labels.insert(0, x.labels.pop(1))
    # x = x.combine((1, 2))
    # fig = plot_channels(x)
