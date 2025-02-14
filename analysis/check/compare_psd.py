from ieeg.viz.ensemble import figure_compare
from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import trial_ieeg
import os
import matplotlib.pyplot as plt

# %% Set up paths
# ------------
HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
TASK = "SentenceRep"
layout = get_data("SentenceRep", root=LAB_root)
subjects = layout.get(return_type="id", target="subject")

# %% Load Data
# ---------
for subject in subjects:
    subject = f"D{29:04}"
    try:
        # raw = raw_from_layout(layout, subject=subject, extension=".edf", desc=None,
        #                       preload=True)
        # clean = raw_from_layout(layout.derivatives["clean"], subject=subject, extension=".edf", desc="clean",
        #                       preload=False)
        notch = raw_from_layout(layout.derivatives["notch"], subject=subject, extension=".edf", desc="notch",
                                preload=False)
        trials = trial_ieeg(notch, "Word", (-0.5, 1), preload=True
                            , reject_by_annotation=False)
        resp = trials["Response"]
        no_resp = trials["Audio"]
        raw = raw_from_layout(layout, subject=subject, extension=".edf", desc=None,
                              preload=False)
        trials = trial_ieeg(raw, ["Response", "Audio"], (-0.5, 1), preload=True
                            , reject_by_annotation=False)
        resp_raw = trials["Response"]
        no_resp_raw = trials["Audio"]
    except Exception as e:
        print(f"Error in {subject}: {e}")
        continue
    figure_compare([resp, no_resp, resp_raw, no_resp_raw],["responses", "audio", "responses raw", "audio raw"],
                   False, n_jobs=1, verbose=30, proj=False, fmax=250, fmin=4,
                   # method="multitaper",
                   # reject_by_annotation=False,
                   # adaptive=False
                   )
    break
    # fig = plt.gcf()
    # fig.savefig(os.path.join(layout.root, "spec", f"{subject}_clean_notch.png"))
    # plt.close(fig)
