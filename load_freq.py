import os
import numpy as np
import scipy.stats as st

from ieeg.io import get_data
from analysis.utils.mat_load import DataLoader
from analysis.grouping import group_elecs
from ieeg.calc.mat import LabeledArray, combine

fpath = os.path.expanduser("~/Box/CoganLab")
layout = get_data('SentenceRep', root=fpath)
conds = {"resp": (-1, 1), "aud_ls": (-0.5, 1.5),
                    "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
                    "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
                    "go_jl": (-0.5, 1.5)}

def load_data(datatype: str, out_type = float):
    loader = DataLoader(layout, conds, datatype, True, 'stats_freq',
                       '.h5')
    zscore = loader.load_dict()
    zscore_ave = combine(zscore, (0, 2))
    for key in zscore_ave.keys():
        for k in zscore_ave[key].keys():
            for f in zscore_ave[key][k].keys():
                zscore_ave[key][k][f] = zscore_ave[key][k][f][:200]
    del zscore
    return LabeledArray.from_dict(zscore_ave, dtype=out_type)

sigs = load_data("significance", out_type=bool)

AUD, SM, PROD, sig_chans = group_elecs(sigs, sigs.labels[1], sigs.labels[0])

pvals = load_data("pval", out_type=float)
data = np.where(pvals > 0.9999, 0.9999, pvals)
zscores = LabeledArray(st.norm.ppf(1 - data), pvals.labels)


