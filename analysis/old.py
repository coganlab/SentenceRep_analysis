def calinski_harabasz_old(X_in: np.ndarray | csr_matrix,
                      W_in: np.ndarray | csr_matrix, n_clusters: int = None
                      ) -> float:
    """
    Here, X is the original data matrix, W and H are the non-negative factor
    matrices from the ONMF decomposition. The function first computes the
    centroid of each cluster defined by the columns of W by taking the mean
    of the data points assigned to that cluster. It then computes the total
    sum of squares (TSS), which is the sum of squared distances between each
    data point and the mean of all data points. It computes the
    between-cluster sum of squares (BSS), which is the sum of squared
    distances between each cluster centroid and the mean of all data points,
    weighted by the number of data points assigned to that cluster. Finally,
    it computes the within-cluster sum of squares (WSS), which is the sum of
    squared distances between each data point and its assigned cluster
    centroid, weighted by the assignment weights in W.
    """

    X = _check_array(X_in)
    W = _check_array(W_in)

    if n_clusters is None:
        n_clusters = W.shape[1]
    n_samples = X.shape[0]

    # Compute the centroid of each cluster
    centroids = np.zeros((n_clusters, X.shape[1]))
    for i in range(n_clusters):
        centroids[i] = np.mean(X[W[:, i] > 0], axis=0)

    # Compute the between-cluster sum of squares
    BSS = np.sum(np.sum(W[:, i, np.newaxis] * pairwise_distances(
        centroids[i, np.newaxis, :], np.mean(X, axis=0, keepdims=True))**2)
                 for i in range(n_clusters))

    # Compute the within-cluster sum of squares
    WSS = np.sum(np.sum(W[:, i, np.newaxis] * pairwise_distances(
        X, centroids[i, np.newaxis, :])**2) for i in range(n_clusters))

    # Compute the CH index
    ch = (BSS / (n_clusters - 1)) / (WSS / (n_samples - n_clusters))

    return ch