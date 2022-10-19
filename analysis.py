from utils.mat_load import get_sigs, load_all, group_elecs
from plotting import plot_opt_k, plot_weight_dist, alt_plot
from utils.calc import ArrayLike, BaseEstimator, stitch_mats
from decomposition import estimate, to_sklearn_dataset, TimeSeriesScalerMinMax, NMF

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame as df

Task, all_sigZ, all_sigA, sig_chans, sigMatChansLoc, sigMatChansName, Subject = load_all('data/whole.mat')

cond = 'LSwords'
aud = all_sigA[cond]["AuditoryWhole"]
go = all_sigA[cond]["GoWhole"]
resp = all_sigA[cond]["ResponseWhole"]
audz = all_sigZ[cond]["AuditoryWhole"]
goz = all_sigZ[cond]["GoWhole"]
respz = all_sigZ[cond]["ResponseWhole"]
newSet = [aud, go, resp]
sigConcat = np.concatenate(newSet, axis=1)
newSet = [aud, go, resp, audz, goz, respz]
sig_chans2 = list(set(np.concatenate(list(sig_chans[cond].values())).ravel().tolist()))
sig_chans2 = [sig - 1 for sig in sig_chans2]
nonActive = np.where(np.all(np.isclose(sigConcat, 0,), axis=1))
newSet.append(sigConcat)
for i, allign in enumerate(newSet):
    newSet[i] = newSet[i][sig_chans2]
[aud, go, resp, audz, goz, respz, sigConcat] = newSet[:]
audwt = np.multiply(aud, audz)
gowt = np.multiply(go, goz)
respwt = np.multiply(resp, respz)
stitched = stitch_mats([audz[:, 0:200], goz[:, 150:350],
                        respz[:, 200:400]], [0,0], axis=1)
# %% Generate the summed signals
sigSum = np.sum(np.array(newSet[0:3]), axis=0)
sigSumWt = np.sum(np.array([audwt,gowt,respwt]), axis=0)
sigSumZ = np.sum(np.array(newSet[3:6]), axis=0)
middle = (sigSum >= 2)[:,50:250]
start = (sigSum >= 1)[:,0:50]
end = (sigSum >= 1)[:,250:400]
checkedSig = np.concatenate([start, middle, end], axis=1)
# %% Data Plotting
# plt.matshow(stitched)
# plt.plot(np.mean(stitched, axis=0))

# %% Grid search decomposition
x = to_sklearn_dataset(TimeSeriesScalerMinMax((0.0001, 1)).fit_transform(sigSumZ))
gridsearch = estimate(x, NMF(max_iter=100000), 2)
res = df(gridsearch.cv_results_)
estimator = gridsearch.best_estimator_
# estimator.n_components = 3
y = estimator.fit_transform(x)
# %% decomposition plotting
plot_weight_dist(x, y)#,["PROD", "SM", "AUD"],["blue","red","lime"])
# plt.legend()
# alt_plot(x,np.argmax(y,1))
# decomp_sigs = np.dot(x.T, y)
# plt.plot(decomp_sigs)
# %% save data
gridsearch.scorer_ = gridsearch.scoring = {}
np.save('data/gridsearch.npy', [gridsearch, x, y], allow_pickle=True)
# plt.plot(decomp_sigs)
plt.savefig('data/decomp.png')
