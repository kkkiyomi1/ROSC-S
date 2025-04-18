import numpy as np
from numpy.linalg import svd


def z_score_normalize(X):
    """
    Perform Z-score normalization on features (axis=0).
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8)


def nuclear_norm(M):
    """
    Compute the nuclear norm (sum of singular values).
    """
    _, s, _ = svd(M, full_matrices=False)
    return np.sum(s)


def frobenius_norm(M):
    """
    Compute the Frobenius norm of a matrix.
    """
    return np.linalg.norm(M, 'fro')


def pairwise_distance(X1, X2, metric='euclidean'):
    """
    Compute pairwise distance between samples.
    """
    from scipy.spatial.distance import cdist
    return cdist(X1, X2, metric=metric)
