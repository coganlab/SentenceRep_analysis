import os

import numpy as np

from ieeg.io import get_data
from analysis.utils.mat_load import load_dict
from ieeg.calc.mat import LabeledArray, combine

fpath = os.path.expanduser("~/Box/CoganLab")
layout = get_data('SentenceRep', root=fpath)
conds = {"resp": (-1, 1), "aud_ls": (-0.5, 1.5),
                    "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
                    "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
                    "go_jl": (-0.5, 1.5)}

zscore = load_dict(layout, conds, "zscore", False, 'stats_freq', '.h5')
zscore_ave = combine(zscore, (0, 3))
del zscore
data = LabeledArray.from_dict(zscore_ave['resp'])
