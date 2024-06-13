from ieeg.viz.ensemble import figure_compare
from ieeg.io import get_data, raw_from_layout
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
    subject = f"D{24:04}"
    try:
        raw = raw_from_layout(layout, subject=subject, extension=".edf", desc=None,
                              preload=True)
        clean = raw_from_layout(layout.derivatives["clean"], subject=subject, extension=".edf", desc="clean",
                              preload=True)
        notch = raw_from_layout(layout.derivatives["notch"], subject=subject, extension=".edf", desc="notch",
                                preload=True)
    except Exception as e:
        print(f"Error in {subject}: {e}")
        continue
    figure_compare([raw, clean, notch],["raw", "clean", "notch"],
                   False, n_jobs=6, verbose=30, proj=True, fmax=250, method="multitaper", reject_by_annotation=False, adaptive=False)
    break
    # fig = plt.gcf()
    # fig.savefig(os.path.join(layout.root, "spec", f"{subject}_clean_notch.png"))
    # plt.close(fig)