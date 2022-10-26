from utils.mat_load import get_sigs, load_all, group_elecs
from plotting import plot_clustering, plot_weight_dist, alt_plot
from utils.calc import ArrayLike, BaseEstimator, stitch_mats
from decomposition import estimate, to_sklearn_dataset, TimeSeriesScalerMinMax, NMF

import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numpy as np
from pandas import DataFrame as df
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.io import loadmat
import tensortools as tt
from tensortools.cpwarp import fit_shifted_cp

Task, all_sigZ, all_sigA, sig_chans, sigMatChansLoc, sigMatChansName, Subject = load_all('data/pydata.mat')
sigMatChansName = sigMatChansName["LSwords"]['AuditorytoResponse']
data = loadmat("data/pydata_3d.mat", simplify_cells=True)
names_3d = data['channelNames']
trialInfo_3d = data["trialInfo"]
listen_speak_3d = data["listenSpeak"]
Subject_3d = data["subj"]
del data
concat_3d = np.concatenate(list(listen_speak_3d.values()), 2)
# good and non white matter channels
all_sig_chans = np.array([], dtype=int)
for cond in sig_chans.values():
    for chans in cond.values():
        all_sig_chans = np.concatenate([all_sig_chans, chans])
all_sig_chans = list(set(all_sig_chans.ravel()))
cond = 'LSwords'
resp = all_sigA[cond]["Response"]
respz = all_sigZ[cond]["Response"]
aud = all_sigA[cond]["AuditorywDelay"][:,:150]
audz = all_sigZ[cond]["AuditorywDelay"][:,:150]
part = all_sigA[cond]["StartPart"]
partz = all_sigZ[cond]["StartPart"]
go = all_sigA[cond]["DelaywGo"]
goz = all_sigZ[cond]["DelaywGo"]
# part = dict(aud=aud, audz=audz, go=go, goz=goz, resp=resp, respz=respz)
# SM, AUD, PROD = group_elecs(all_sigA, sig_chans)

# Task, all_sigZ, all_sigA, sig_chans, sigMatChansLoc, sigMatChansName, Subject = load_all('data/whole.mat')
# resp = all_sigA[cond]["ResponseWhole"]
# respz = all_sigZ[cond]["ResponseWhole"]
# aud = all_sigA[cond]["AuditoryWhole"]
# audz = all_sigZ[cond]["AuditoryWhole"]
# go = all_sigA[cond]["GoWhole"]
# goz = all_sigZ[cond]["GoWhole"]

sigConcat = np.concatenate([part,aud, go, resp], axis=1)
newSet = [aud, go, resp, audz, goz, respz, sigConcat, sigMatChansName]
active = np.where(np.any(sigConcat == 1, axis=1))[0]
# sig_chans2 = np.intersect1d(active,all_sig_chans)
# sig_chans3 = np.concatenate([SM, AUD, PROD])
for i, allign in enumerate(newSet): #active channels during condition
    newSet[i] = newSet[i][active]
[aud, go, resp, audz, goz, respz, sigConcat, sigMatChansName] = newSet[:]
data_3d = concat_3d[np.isin(names_3d,sigMatChansName),:,:]
audwt = np.multiply(aud, audz)
gowt = np.multiply(go, goz)
respwt = np.multiply(resp, respz)
# %% Generate the summed signals
# sigSum = np.sum(np.array(newSet[0:3]), axis=0)
# sigSumWt = np.sum(np.array([audwt, gowt, respwt]), axis=0)
# sigSumZ = np.sum(np.array(newSet[3:6]), axis=0)
# %% 3d nmf
# Fit an ensemble of models, 4 random replicates / optimization runs per model rank
# ensemble = tt.Ensemble(fit_method="ncp_hals")
# ensemble.fit(data_3d, ranks=range(1, 6), replicates=4)
# varience = 1-np.mean(np.array([np.square(np.array(ensemble.objectives(i))) for i in range(1,6)]),1)
# fig, axes = plt.subplots(1, 2)
# tt.plot_objective(ensemble, ax=axes[0])   # plot reconstruction error as a function of num components.
# tt.plot_similarity(ensemble, ax=axes[1])  # plot model similarity as a function of num components.
# fig.tight_layout()

# Plot the low-d factors for an example model, e.g. rank-2, first optimization run / replicate.
# num_components = 2
# replicate = 0
# tt.plot_factors(ensemble.factors(num_components)[replicate])
# plt.show()# plot the low-d factors

# Time shifted
# rot_data = np.rot90(data_3d,axes=(0,2)) # time X trials X chans now
data_nonneg = data_3d - np.min(data_3d)
# fit = []
# sim = []
# for rank in range(1,6):
#     fit.append(fit_shifted_cp(data_3d,rank=rank, max_shift_axis1=0.0000001,))
#     if rank == 1:
#         sim.append(1)
#     else:
#         sim.append(tt.kruskal_align(fit[-1].factors,fit[-2].factors, permute_U=True, permute_V=True))
results = Parallel(-1, verbose=0)(delayed(fit_shifted_cp)(
    data_nonneg.copy(), i,n_restarts=5,max_shift_axis1=15, max_iter=10000) for i in range(1,9))
# %% Generate the stitched signals
# stitched = stitch_mats([aud, go, resp], [0, 0], axis=1)
# stitchedz = stitch_mats([audz, goz, respz], [0, 0], axis=1)
# stitchedwt = stitch_mats([audwt, gowt, respwt], [0, 0], axis=1)
# # %% Data Plotting
# # plt.matshow(sigSum)
# # plt.plot(np.mean(stitched, axis=0))
#
# # %% Grid search decomposition
# x = to_sklearn_dataset(TimeSeriesScalerMinMax((0, 1)).fit_transform(stitched))
# gridsearch = estimate(x, NMF(max_iter=100000), 2)
# res = df(gridsearch.cv_results_)
# estimator = gridsearch.best_estimator_
# # estimator.n_components = 3
# y = estimator.fit_transform(x)
# # %% decomposition plotting
# plot_weight_dist(x, y, ["PROD", "SM", "AUD"],["blue","red","lime"])
# plt.legend()
# # prod = sig_chans3[np.argmax(y,1)==0]
# # aud = sig_chans3[np.argmax(y,1)==1]
# # sm = sig_chans3[np.argmax(y,1)==2]
# # y_true = np.concatenate([[0]*len(PROD), [1]*len(AUD), [2]*len(SM)])
# y_true = [0 if chan in PROD else 2 if chan in AUD else 1 for chan in sig_chans3]
# ConfusionMatrixDisplay.from_predictions(y_true,np.argmax(y,1),normalize='true',display_labels=["PROD", "AUD", "SM"])
# # alt_plot(x,np.argmax(y,1))
# # decomp_sigs = np.dot(x.T, y)
# # plt.plot(decomp_sigs)
# # %% save data
# gridsearch.scorer_ = gridsearch.scoring = {}
# np.save('data/gridsearch_stack2.npy', [gridsearch, x, y], allow_pickle=True)
# # plt.plot(decomp_sigs)
# plt.savefig('data/decomp_stack2.png')
# # %% load data
# # new, x, y = np.load('data/gridsearch_stitchedwt.npy', allow_pickle=True)[:]
# # res = df(new.cv_results_)
# # estimator = new.best_estimator_
