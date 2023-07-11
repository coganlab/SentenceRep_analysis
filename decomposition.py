import os
import numpy as np
from ieeg.io import get_data
from ieeg.viz.utils import plot_dist
from ieeg.calc.utils import stitch_mats
import matplotlib.pyplot as plt
from utils.mat_load import load_intermediates, group_elecs
from sklearn.decomposition import NMF


def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    from numpy import eye, asarray, dot, sum, diag
    from scipy.linalg import svd
    p, k = Phi.shape
    R = eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u, s, vh = svd(dot(Phi.T, asarray(Lambda) ** 3 - (gamma / p) * dot(Lambda, diag(diag(dot(Lambda.T, Lambda))))))
        R = dot(u, vh)
        d = sum(s)
        if d_old != 0 and d / d_old < 1 + tol: break
    return dot(Phi, R)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ## check if currently running a slurm job
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

    ## Load the data
    epochs, all_power, names = load_intermediates(layout, conds, "power")
    signif, all_sig, _ = load_intermediates(layout, conds, "significance")

    ## plot significant channels
    AUD, SM, PROD, sig_chans = group_elecs(all_sig, names, conds)
    # %%
    aud_c = "aud_ls"
    go_c = "go_ls"
    resp_c = "resp"
    stitch_aud = stitch_mats([all_power[aud_c][AUD, :150],
                              all_power[go_c][AUD, :]], [0], axis=1)
    stitch_sm = stitch_mats([all_power[aud_c][SM, :150],
                             all_power[go_c][SM, :]], [0], axis=1)
    stitch_prod = stitch_mats([all_power[aud_c][PROD, :150],
                               all_power[go_c][PROD, :]], [0], axis=1)
    stitch_all = np.vstack([stitch_aud, stitch_sm, stitch_prod])
    labels = np.concatenate([np.ones([len(AUD)]), np.ones([len(SM)]) * 2,
                             np.ones([len(PROD)]) * 3])
    # %%

    import nimfa

    bmf = nimfa.Bmf(stitched, seed="nndsvd", rank=4, max_iter=1000, lambda_w=1.01, lambda_h=1.01)
    bmf_fit = bmf()
    n = bmf.estimate_rank([2,3,4,5,6,7,8],n_run=100)
    from MEPONMF.onmf_DA import DA
    from MEPONMF.onmf_DA import ONMF_DA
    # k = 10
    # param = dict(tol=1e-8, alpha=1.002,
    #            purturb=0.5, verbos=1, normalize=False)
    # W, H, model = ONMF_DA.func(stitched, k=k, **param, auto_weighting=False)
    # model.plot_criticals(log=True)
    # plt.show()
    # k = model.return_true_number()
    # W, H, model2 = ONMF_DA.func(stitched, k=k, **param, auto_weighting=True)
    # model2 = DA(**param,K=k, max_iter=1000)
    # model2.fit(stitched,Px='auto')
    # Y,P = model2.cluster()
    # model2.plot_criticals(log=True)
    # plt.show()
    # plot_weight_dist(stitchedz, Y)
    # model2.pie_chart()

    # for k in np.array(range(7))+1:
    #     W, H, model = ONMF_DA.func(stitched, k, alpha=1.05, purturb=0.01, tol=1e-7)
    #     # model = DA(k, tol=1e-4, max_iter=1000, alpha=1.05,
    #     #             purturb=0.01, beta_final=None, verbos=0, normalize=False)
    #     # model.fit(stitched, Px='auto')
    #     # y, P = model.cluster()
    #     cost.append(model.return_cost())
    # x = to_sklearn_dataset(TimeSeriesScalerMinMax((0, 1)).fit_transform(stitched))
    # gridsearch = estimate(x, NMF(max_iter=100000, tol=1e-8), 3)
    # estimator = gridsearch.best_estimator_
    # estimator = NMF(max_iter=100000,init='nndsvda',alpha_W=0.01,
    #                              alpha_H=0.5, verbose=2)
    # estimator = FactorAnalysis(max_iter=100000,copy=True)
    # test = np.linspace(0, 1, 5)
    # # param_grid = {'n_components': [3], 'init': ['nndsvda'],
    # #               'solver': ['mu'], 'beta_loss': [2, 1, 0.5], 'l1_ratio': test,
    # #               'alpha_W': [0], 'alpha_H': test}
    # param_grid = {'n_components': [4],'rotation' : ['varimax', 'quartimax']}
    # gridsearch = estimate(to_sklearn_dataset(TimeSeriesScalerMinMax((0, 1)).fit_transform(stitched)), estimator,
    #                       param_grid, 5)
    # estimator = gridsearch.best_estimator_
    # estimator.n_components = 4
    # y = estimator.fit_transform(to_sklearn_dataset(TimeSeriesScalerMinMax((0, 1)).fit_transform(stitched)))
    # res = df(gridsearch.cv_results_)
    # decomp = td.non_negative_parafac
    # tens = td.CP_NN_HALS(3, n_iter_max=10000, init='random', exact=True, tol=1e-7)
    # tens.mask = tl.tensor(stitched)
    # tens.fit(tl.tensor(stitchedz))
    # y = tens.decomposition_.factors[0]
    #k = 4
    #W, H, model = ONMF_DA.func(stitched, k, alpha=1.05, purturb=0.01, tol=1e-4)
    #plot_weight_dist(stitched, W)

    #
    # gridsearch.scorer_ = gridsearch.scoring = {}
    # np.save('data/gridsearch.npy', [gridsearch, x, y], allow_pickle=True)
    # plt.plot(decomp_sigs)
    # plt.savefig('data/decomp.png')
