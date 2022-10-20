from utils.mat_load import get_sigs, load_all, group_elecs
from plotting import plot_clustering, plot_weight_dist, alt_plot
from utils.calc import ArrayLike, BaseEstimator, stitch_mats
from decomposition import estimate, to_sklearn_dataset, TimeSeriesScalerMinMax, NMF

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame as df

Task, all_sigZ, all_sigA, sig_chans, sigMatChansLoc, sigMatChansName, Subject = load_all('data/whole.mat')

cond = 'LSwords'
aud = all_sigA[cond]["AuditoryWhole"]
go = all_sigA[cond]["GoWhole"]
audz = all_sigZ[cond]["AuditoryWhole"]
goz = all_sigZ[cond]["GoWhole"]
resp = all_sigA[cond]["ResponseWhole"]
respz = all_sigZ[cond]["ResponseWhole"]
sigConcat = np.concatenate([aud[:, 0:150],
                            go[:, 150:400],
                            resp[:, 150:400]], axis=1)
newSet = [aud, go, resp, audz, goz, respz]
sig_chans2 = list(set(np.concatenate(list(sig_chans[cond].values())).ravel().tolist()))
sig_chans2 = [sig - 1 for sig in sig_chans2]
nonActive = np.where(np.all(np.isclose(sigConcat, 0,), axis=1))
newSet.append(sigConcat)
for i, allign in enumerate(newSet):
    newSet[i] = newSet[i][np.any(sigConcat==1, axis=1)]
[aud, go, resp, audz, goz, respz, sigConcat] = newSet[:]
audwt = np.multiply(aud, audz)
gowt = np.multiply(go, goz)
respwt = np.multiply(resp, respz)
stitched = stitch_mats([aud[:, 0:150], go[:, 150:400],
                        resp[:, 150:400]], [0,0], axis=1)
stitchedz = stitch_mats([audz[:, 0:150], goz[:, 150:400],
                            respz[:, 150:400]], [0,0], axis=1)
stitchedwt = stitch_mats([audwt[:, 0:150], gowt[:, 150:400],
                            respwt[:, 150:400]], [0,0], axis=1)
# %% Generate the summed signals
sigSum = np.sum(np.array(newSet[0:3]), axis=0)
sigSumWt = np.sum(np.array([audwt,gowt,respwt]), axis=0)
sigSumZ = np.sum(np.array(newSet[3:6]), axis=0)
# %% Data Plotting
# plt.matshow(stitched)
# plt.plot(np.mean(stitched, axis=0))

# %% Grid search decomposition
x = to_sklearn_dataset(TimeSeriesScalerMinMax((0, 3)).fit_transform(stitched))
gridsearch = estimate(x, NMF(max_iter=100000), 2)
res = df(gridsearch.cv_results_)
estimator = gridsearch.best_estimator_
#estimator.n_components = 3
y = estimator.transform(x)
# %% decomposition plotting
plot_weight_dist(x, y,["PROD", "SM", "AUD"],["blue","red","lime"])
# plt.legend()
# alt_plot(x,np.argmax(y,1))
# decomp_sigs = np.dot(x.T, y)
# plt.plot(decomp_sigs)
# %% save data
gridsearch.scorer_ = gridsearch.scoring = {}
np.save('data/gridsearch.npy', [gridsearch, x, y], allow_pickle=True)
# plt.plot(decomp_sigs)
plt.savefig('data/decomp.png')
