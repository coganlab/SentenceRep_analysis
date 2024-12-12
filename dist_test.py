import ieeg.viz
from scipy.stats import permutation_test, boxcox
import numpy as np
import matplotlib.pyplot as plt

def mean_diff(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

@np.vectorize(signature='(n)->(n)')
def cox(x):
    x_min = np.min(x) - 0.001
    xt, lam = boxcox(x - x_min)
    x_min = (abs(x_min) ** lam) * -1 if x_min < 0 else x_min ** lam
    return xt + x_min

# allocate memory for two normal distributions
n = 2000 # number of samples
m = 1000 # number of vectorized iterations
rng = np.random.default_rng()
arr1 = rng.normal(0, 1, (n, m))

# set up permutation test parameters
kwargs = {'n_resamples': 1000, 'statistic': mean_diff, 'batch': 1000,
          'alternative': 'greater', 'vectorized': True, 'axis': 0,
          'random_state': rng}
samplings = ((1000, 1000), (500, 1500), (250, 1750), (100, 1900))
fig, axss = plt.subplots(3, 4)
stds = (0.5, 1, 2)

# iterate over the standard deviations
for i, axs in zip(stds, axss):

    # group 2 has different std
    arr2 = rng.normal(0, i, (n, m))

    # iterate over the sampling proportions
    for (prop1, prop2), ax in zip(samplings, axs):

        # combine the distributions and boxcox transform
        data = np.concatenate([arr2[:prop1], arr1[:prop2]], axis=0)
        data_fixed = cox(data)
        inputs = [data_fixed[:prop1], data_fixed[prop1:]]

        # run the permutation test
        res = permutation_test(inputs, **kwargs).pvalue

        # plot the histogram of p-values
        ax.hist(res, bins=20)
        ax.set_ylim(0, 200)
        if prop1 == 1000:
            ax.set_ylabel(f'Count (grp2 std: {i})')
        else:
            ax.set_yticklabels([])
        if i == 2:
            ax.set_xlabel('p-value')
        else:
            if i == 0.5:
                ax.set_title(f'{prop1}/{prop2}')
            ax.set_xticklabels([])
fig.suptitle("Effect of grp2/grp1 sampling and variance on perm test p-values")
fig.show()
