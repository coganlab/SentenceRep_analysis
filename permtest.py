import numpy as np
from ieeg.arrays.reshape import windower
import inspect
from itertools import combinations
from scipy.special import comb, boxcox as _boxcox
from functools import partial
from scipy.stats import boxcox_normmax, rankdata

AxisError: type[Exception]

def brunnermunzel(x: np.ndarray, y: np.ndarray, axis=None, nan_policy='omit'):
    """Compute the Brunner-Munzel test on samples x and y.

    The Brunner-Munzel test is a nonparametric test of the null hypothesis that
    when values are taken one by one from each group, the probabilities of
    getting large values in both groups are equal.
    Unlike the Wilcoxon-Mann-Whitney's U test, this does not require the
    assumption of equivariance of two groups. Note that this does not assume
    the distributions are same. This test works on two independent samples,
    which may have different sizes.

    Parameters
    ----------
    x, y : array_like
        Array of samples, should be one-dimensional.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

          * 'two-sided'
          * 'less': one-sided
          * 'greater': one-sided
    distribution : {'t', 'normal'}, optional
        Defines how to get the p-value.
        The following options are available (default is 't'):

          * 't': get the p-value by t-distribution
          * 'normal': get the p-value by standard normal distribution.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    Returns
    -------
    statistic : float
        The Brunner-Munzer W statistic.
    pvalue : float
        p-value assuming an t distribution. One-sided or
        two-sided, depending on the choice of `alternative` and `distribution`.

    See Also
    --------
    mannwhitneyu : Mann-Whitney rank test on two samples.

    Notes
    -----
    Brunner and Munzel recommended to estimate the p-value by t-distribution
    when the size of data is 50 or less. If the size is lower than 10, it would
    be better to use permuted Brunner Munzel test (see [2]_).

    References
    ----------
    .. [1] Brunner, E. and Munzel, U. "The nonparametric Benhrens-Fisher
           problem: Asymptotic theory and a small-sample approximation".
           Biometrical Journal. Vol. 42(2000): 17-25.
    .. [2] Neubert, K. and Brunner, E. "A studentized permutation test for the
           non-parametric Behrens-Fisher problem". Computational Statistics and
           Data Analysis. Vol. 51(2007): 5192-5204.

    Examples
    --------
    >>> from scipy.stats import brunnermunzel as bz
    >>> x1 = np.array([1,2,1,1,1,1,1,1,1,1,2,4,1,1])
    >>> x2 = np.array([3,3,4,3,1,2,3,1,1,5,4])
    >>> brunnermunzel(x1, x2), bz(x1, x2, alternative='greater').statistic
    3.1374674823029505
    >>> x3 = np.array([[1,2,1,1],[1,1,1,1],[1,1,2,4]])
    >>> x4 = np.array([[3,3,4,3],[1,2,3,1], [1,5,4,4]])
    >>> brunnermunzel(x3, x4, axis=0), bz(x3, x4, axis=0, alternative='greater').statistic
    3.1374674823029505
    >>> brunnermunzel(x3, x4, axis=1), bz(x3, x4, axis=1, alternative='greater').statistic
    >>> brunnermunzel(x3, x4, axis=None), bz(x3, x4, axis=None, alternative='greater').statistic
    """

    if axis is None:
        nx, ny = x.size, y.size
        idxx = slice(0, nx)
        idxy = slice(nx, nx+ny)
        x, y = x.flat, y.flat
        concat = np.concatenate((x, y), axis=0)
    else:
        while axis < 0:
            axis += x.ndim
        nx, ny = x.shape[axis], y.shape[axis]
        idxx = tuple(slice(None) if i != axis else slice(0, nx)
                 for i in range(x.ndim))
        idxy = tuple(slice(None) if i != axis else slice(nx, nx+ny)
                    for i in range(x.ndim))
        concat = np.concatenate((x, y), axis=axis)

    where = ~np.isnan(concat)
    if nan_policy == 'omit':
        rank = partial(rankdata, nan_policy=nan_policy)
        wherex, wherey = where[idxx], where[idxy]
    else:
        rank = rankdata
        wherex = wherey = None
        if np.any(~where) and nan_policy == 'raise':
            raise ValueError("The input contains NaN.")

    kwargsx = dict(axis=axis, where=wherex, keepdims=True)
    kwargsy = dict(axis=axis, where=wherey, keepdims=True)

    rankc = rank(concat, axis=axis)
    rankcx, rankcy = rankc[idxx], rankc[idxy]
    rankcx_mean, rankcy_mean = rankcx.mean(**kwargsx), rankcy.mean(**kwargsy)
    rankx, ranky = rank(x, axis=axis), rank(y, axis=axis)
    rankx_mean, ranky_mean = rankx.mean(**kwargsx), ranky.mean(**kwargsy)

    Sx = np.sum(np.power(rankcx - rankx - rankcx_mean + rankx_mean, 2.0),
                **kwargsx) / (nx - 1)
    Sy = np.sum(np.power(rankcy - ranky - rankcy_mean + ranky_mean, 2.0),
                **kwargsy) / (ny - 1)

    wbfn = nx * ny * (rankcy_mean - rankcx_mean)
    wbfn /= (nx + ny) * np.sqrt(nx * Sx + ny * Sy)
    return np.squeeze(wbfn)

# @np.frompyfunc(nin=2, nout=1)
@np.vectorize(signature="(n),(m)->()",
              otypes=[float])
def brunner(x, y):
    nx = len(x)
    ny = len(y)

    rankc = rankdata(np.concatenate((x, y)))
    rankcx = rankc[0:nx]
    rankcy = rankc[nx:nx+ny]
    rankcx_mean = np.mean(rankcx)
    rankcy_mean = np.mean(rankcy)
    rankx = rankdata(x)
    ranky = rankdata(y)
    rankx_mean = np.mean(rankx)
    ranky_mean = np.mean(ranky)

    Sx = np.sum(np.power(rankcx - rankx - rankcx_mean + rankx_mean, 2.0))
    Sx /= nx - 1
    Sy = np.sum(np.power(rankcy - ranky - rankcy_mean + ranky_mean, 2.0))
    Sy /= ny - 1

    wbfn = nx * ny * (rankcy_mean - rankcx_mean)
    wbfn /= (nx + ny) * np.sqrt(nx * Sx + ny * Sy)
    return wbfn

def permutation_test(data, statistic, *, permutation_type='independent',
                     vectorized=None, n_resamples=9999, batch=None,
                     alternative="two-sided", axis=0, rng=None,
                     do_boxcox=False, null_sampling='same'):

    if do_boxcox:
        data_in = _cox(*data, axis=axis)
    else:
        data_in = data
    args = _permutation_test_iv(data_in, statistic, permutation_type, vectorized,
                                n_resamples, batch, alternative, axis,
                                rng)
    (data_transformed, statistic, permutation_type, vectorized, n_resamples, batch,
     alternative, axis, rng) = args

    observed = statistic(*data_transformed, axis=-1)
    # data_transformed, observed = boxcox_trans(*data_transformed, obs_diff=observed)
    null_calculators = {"independent": _calculate_null_both}
    null_calculator_args = (data_transformed, statistic, n_resamples,
                            batch, rng, null_sampling)
    calculate_null = null_calculators[permutation_type]
    null_distribution, n_resamples, exact_test = (
        calculate_null(*null_calculator_args))

    # See References [2] and [3]
    adjustment = 0 if exact_test else 1

    # relative tolerance for detecting numerically distinct but
    # theoretically equal values in the null distribution
    eps =  (0 if not np.issubdtype(observed.dtype, np.inexact)
            else np.finfo(observed.dtype).eps*100)
    gamma = np.abs(eps * observed)

    def less(null_distribution, observed):
        cmps = null_distribution <= observed + gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)
        return pvalues

    def greater(null_distribution, observed):
        cmps = null_distribution >= observed - gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)
        return pvalues

    def two_sided(null_distribution, observed):
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues

    compare = {"less": less,
               "greater": greater,
               "two-sided": two_sided}

    pvalues = compare[alternative](null_distribution, observed)
    pvalues = np.clip(pvalues, 0, 1)

    return observed, pvalues, null_distribution

def _permutation_test_iv(data, statistic, permutation_type, vectorized,
                         n_resamples, batch, alternative, axis, rng):
    """Input validation for `permutation_test`."""

    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError("`axis` must be an integer.")

    permutation_types = {'samples', 'pairings', 'independent'}
    permutation_type = permutation_type.lower()
    if permutation_type not in permutation_types:
        raise ValueError(f"`permutation_type` must be in {permutation_types}.")

    if vectorized not in {True, False, None}:
        raise ValueError("`vectorized` must be `True`, `False`, or `None`.")

    if vectorized is None:
        vectorized = 'axis' in inspect.signature(statistic).parameters

    if not vectorized:
        statistic = _vectorize_statistic(statistic)

    message = "`data` must be a tuple containing at least two samples"
    try:
        if len(data) < 2 and permutation_type == 'independent':
            raise ValueError(message)
    except TypeError:
        raise TypeError(message)

    data = _broadcast_arrays(data, axis)
    data_iv = []
    for sample in data:
        sample = np.atleast_1d(sample)
        if sample.shape[axis] <= 1:
            raise ValueError("each sample in `data` must contain two or more "
                             "observations along `axis`.")
        sample = np.moveaxis(sample, axis_int, -1)
        data_iv.append(sample)

    n_resamples_int = (int(n_resamples) if not np.isinf(n_resamples)
                       else np.inf)
    if n_resamples != n_resamples_int or n_resamples_int <= 0:
        raise ValueError("`n_resamples` must be a positive integer.")

    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError("`batch` must be a positive integer or None.")

    alternatives = {'two-sided', 'greater', 'less'}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f"`alternative` must be in {alternatives}")

    if rng is None or isinstance(rng, int):
        rng = np.random.RandomState(rng)
    elif not hasattr(rng, 'permutation'):
        raise ValueError("`rng` must have a `permutation`"
                           " method or be an integer.")

    return (data_iv, statistic, permutation_type, vectorized, n_resamples_int,
            batch_iv, alternative, axis_int, rng)

def _batch_generator(iterable, batch):
    """A generator that yields batches of elements from an iterable"""
    iterator = iter(iterable)
    if batch <= 0:
        raise ValueError("`batch` must be positive.")
    z = [item for i, item in zip(range(batch), iterator)]
    while z:  # we don't want StopIteration without yielding an empty list
        yield z
        z = [item for i, item in zip(range(batch), iterator)]

def _vectorize_statistic(statistic):
    """Vectorize an n-sample statistic"""
    # This is a little cleaner than np.nditer at the expense of some data
    # copying: concatenate samples together, then use np.apply_along_axis
    def stat_nd(*data, axis=0):
        lengths = [sample.shape[axis] for sample in data]
        split_indices = np.cumsum(lengths)[:-1]
        z = _broadcast_concatenate(data, axis)

        # move working axis to position 0 so that new dimensions in the output
        # of `statistic` are _prepended_. ("This axis is removed, and replaced
        # with new dimensions...")
        z = np.moveaxis(z, axis, 0)

        def stat_1d(z):
            data = np.split(z, split_indices)
            return statistic(*data)

        return np.apply_along_axis(stat_1d, 0, z)[()]
    return stat_nd

def _broadcast_concatenate(arrays, axis, paired=False):
    """Concatenate arrays along an axis with broadcasting."""
    arrays = _broadcast_arrays(arrays, axis if not paired else None)
    res = np.concatenate(arrays, axis=axis)
    return res

def _broadcast_arrays(arrays, axis=None, xp=None):
    """
    Broadcast shapes of arrays, ignoring incompatibility of specified axes
    """
    if not arrays:
        return arrays
    xp, is_comp = get_namespace(*arrays) if xp is None else xp
    xp = np if not is_comp else xp
    arrays = [xp.asarray(arr) for arr in arrays]
    shapes = [arr.shape for arr in arrays]
    new_shapes = _broadcast_shapes(shapes, axis)
    if axis is None:
        new_shapes = [new_shapes]*len(arrays)
    return [xp.broadcast_to(array, new_shape)
            for array, new_shape in zip(arrays, new_shapes)]

def _broadcast_shapes(shapes, axis=None):
    """
    Broadcast shapes, ignoring incompatibility of specified axes
    """
    if not shapes:
        return shapes

    # input validation
    if axis is not None:
        axis = np.atleast_1d(axis)
        message = '`axis` must be an integer, a tuple of integers, or `None`.'
        try:
            with np.errstate(invalid='ignore'):
                axis_int = axis.astype(int)
        except ValueError as e:
            raise AxisError(message) from e
        if not np.array_equal(axis_int, axis):
            raise AxisError(message)
        axis = axis_int

    # First, ensure all shapes have same number of dimensions by prepending 1s.
    n_dims = max([len(shape) for shape in shapes])
    new_shapes = np.ones((len(shapes), n_dims), dtype=int)
    for row, shape in zip(new_shapes, shapes):
        row[len(row)-len(shape):] = shape  # can't use negative indices (-0:)

    # Remove the shape elements of the axes to be ignored, but remember them.
    if axis is not None:
        axis[axis < 0] = n_dims + axis[axis < 0]
        axis = np.sort(axis)
        if axis[-1] >= n_dims or axis[0] < 0:
            message = (f"`axis` is out of bounds "
                       f"for array of dimension {n_dims}")
            raise AxisError(message)

        if len(np.unique(axis)) != len(axis):
            raise AxisError("`axis` must contain only distinct elements")

        removed_shapes = new_shapes[:, axis]
        new_shapes = np.delete(new_shapes, axis, axis=1)

    # If arrays are broadcastable, shape elements that are 1 may be replaced
    # with a corresponding non-1 shape element. Assuming arrays are
    # broadcastable, that final shape element can be found with:
    new_shape = np.max(new_shapes, axis=0)
    # except in case of an empty array:
    new_shape *= new_shapes.all(axis=0)

    # Among all arrays, there can only be one unique non-1 shape element.
    # Therefore, if any non-1 shape element does not match what we found
    # above, the arrays must not be broadcastable after all.
    if np.any(~((new_shapes == 1) | (new_shapes == new_shape))):
        raise ValueError("Array shapes are incompatible for broadcasting.")

    if axis is not None:
        # Add back the shape elements that were ignored
        new_axis = axis - np.arange(len(axis))
        new_shapes = [tuple(np.insert(new_shape, new_axis, removed_shape))
                      for removed_shape in removed_shapes]
        return new_shapes
    else:
        return tuple(new_shape)

def _calculate_null_both(data, statistic, n_permutations, batch,
                         rng=None, samp_opt='sub'):
    """
    Calculate null distribution for independent sample tests.
    """
    n_samples = len(data)

    # compute number of permutations
    # (distinct partitions of data into samples of these sizes)
    n_obs_i = [sample.shape[-1] for sample in data]  # observations per sample
    n_obs_ic = np.cumsum(n_obs_i)
    n_max = np.prod([comb(n_obs_ic[i], n_obs_ic[i-1])
                     for i in range(n_samples-1, 0, -1)])

    # perm_generator is an iterator that produces permutations of indices
    # from 0 to n_obs. We'll concatenate the samples, use these indices to
    # permute the data, then split the samples apart again.
    def _same():
        return (np.arange(n_obs_i[i]) + (n_obs_ic[i-1] if i > 0 else 0)
                for i in range(n_samples))
    def _over():
        return (rng.choice(n_obs_i[i], max(n_obs_i), shuffle=False,
                           replace=True) + (n_obs_ic[i-1] if i > 0 else 0)
                for i in range(n_samples))
    def _sub():
        return (rng.choice(n_obs_i[i], min(n_obs_i), shuffle=False,
                           replace=False) + (n_obs_ic[i-1] if i > 0 else 0)
                for i in range(n_samples))
    if n_permutations >= n_max:
        exact_test = True
        n_permutations = n_max
        perm_generator = _all_partitions_concatenated(n_obs_i)
    else:
        exact_test = False
        # generate the random indicies for all permutations, either samping
        # each group at the same rate, sub-sampling the larger group,
        # or oversampling the smaller group
        samplings = {'same': _same, 'over': _over, 'sub': _sub}
        # sample_gen = (s + (n_obs_ic[i-1] if i > 0 else 0)
        #               for i, s in enumerate(samplings[samp_opt]))
        sample_gen = samplings[samp_opt]
        perm_generator = rng.permuted(np.stack([np.concatenate(
            list(sample_gen())) for _ in range(n_permutations)]), axis=1)

    batch = batch or int(n_permutations)
    batch_generator = windower(perm_generator, batch, 0, 1)[::batch]
    null_distribution = []

    # First, concatenate all the samples. In batches, permute samples with
    # indices produced by the `perm_generator`, split them into new samples of
    # the original sizes, compute the statistic for each batch, and add these
    # statistic values to the null distribution.
    data = np.concatenate(data, axis=-1)
    for indices in batch_generator:
        # indices = np.array(indices)

        # `indices` is 2D: each row is a permutation of the indices.
        # We use it to index `data` along its last axis, which corresponds
        # with observations.
        # After indexing, the second to last axis of `data_batch` corresponds
        # with permutations, and the last axis corresponds with observations.
        data_batch = data[..., indices]

        # Move the permutation axis to the front: we'll concatenate a list
        # of batched statistic values along this zeroth axis to form the
        # null distribution.
        data_batch = np.moveaxis(data_batch, -2, 0)
        data_batch = np.split(data_batch, n_obs_ic[:-1], axis=-1)
        # data_batch = np.split(data_batch, len(n_obs_ic), axis=-1)
        null_distribution.append(statistic(*data_batch, axis=-1))
    null_distribution = np.concatenate(null_distribution, axis=0)

    return null_distribution, n_permutations, exact_test

def _all_partitions_concatenated(ns):
    """
    Generate all partitions of indices of groups of given sizes, concatenated

    `ns` is an iterable of ints.
    """
    def all_partitions(z, n):
        for c in combinations(z, n):
            x0 = set(c)
            x1 = z - x0
            yield [x0, x1]

    def all_partitions_n(z, ns):
        if len(ns) == 0:
            yield [z]
            return
        for c in all_partitions(z, ns[0]):
            for d in all_partitions_n(c[1], ns[1:]):
                yield c[0:1] + d

    z = set(range(np.sum(ns)))
    for partitioning in all_partitions_n(z, ns[:]):
        x = np.concatenate([list(partition)
                            for partition in partitioning]).astype(int)
        yield x

def _cox(*arrs, axis=0) -> tuple[np.ndarray]:
    """Apply the Box-Cox transformation to the array."""
    arr = np.concatenate(arrs, axis)
    edges = np.cumsum([a.shape[axis] for a in arrs])[:-1]
    if amin := arr.min() < 0:
        arr -= amin - 0.01

    lam = boxcox_normmax(arr[~np.isnan(arr)].flat, method='mle')
    arr_out = _boxcox(arr, lam)
    return np.split(arr_out, edges, axis)


if __name__ == '__main__':

    import ieeg.viz
    import matplotlib.pyplot as plt
    # from ieeg.calc.fast import mean_diff
    # allocate memory for two normal distributions
    n = 2000 # number of samples
    m = 2000 # number of vectorized iterations
    rng = np.random.default_rng()
    arr1 = rng.normal(10, 1, (n, m))

    # set up permutation test parameters
    kwargs = {'n_resamples': 1000, 'statistic': brunnermunzel, 'batch': 100,
              'alternative': 'greater', 'vectorized': True, 'axis': 0,
              'rng': rng}
    samplings = ((100, 100), (100, 250), (100, 500))
    fig, axss = plt.subplots(3, 3)
    stds = (0.5, 1, 2)

    # iterate over the standard deviations
    for i, axs in zip(stds, axss):

        # group 2 has different std
        arr2 = rng.normal(10, i, (n, m))

        # iterate over the sampling proportions
        for (prop1, prop2), ax in zip(samplings, axs):

            # combine the distributions and boxcox transform

            data = np.concatenate([arr2[:prop1], arr1[:prop2]], axis=0)
            # inputs = boxcox_trans(arr2[:prop1], arr1[:prop2], axis=0)[0]
            # temp = (data - data.min() + 0.001).flat
            # lam = boxcox_normmax(temp, method='mle')
            # print(lam)
            # data.flat = _boxcox(temp, lam)
            # data_fixed = cox(data)
            inputs = [data[:prop1], data[prop1:]]

            # run the permutation test
            res = permutation_test(inputs, **kwargs)[1]

            # plot the histogram of p-values
            ax.hist(res, bins=20)
            ax.set_ylim(0, 200)
            if prop2 == 100:
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
