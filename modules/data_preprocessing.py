import numpy as np
import scipy
from scipy.spatial.distance import cdist
from modules.utils import z_score_normalize


def compute_similarity_matrix(X, metric='rbf', sigma=1.0, k=5, normalized=False):
    """
    Compute similarity matrix for a single view.
    
    Parameters:
    -----------
    X : np.ndarray
        Input feature matrix of shape (features, samples)
    metric : str
        Similarity metric: ['rbf', 'cosine', 'knn']
    sigma : float
        RBF kernel width
    k : int
        K for k-nearest neighbors if metric is 'knn'
    normalized : bool
        Whether to normalize similarity matrix

    Returns:
    --------
    S : np.ndarray
        Similarity matrix (n x n)
    """
    X = z_score_normalize(X.T)  # shape (n_samples, n_features)
    if metric == 'rbf':
        dist = cdist(X, X, 'euclidean')
        S = np.exp(-dist**2 / (2 * sigma ** 2))
    elif metric == 'cosine':
        dot = np.dot(X, X.T)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        S = dot / (norms @ norms.T + 1e-8)
    elif metric == 'knn':
        dist = cdist(X, X, 'euclidean')
        S = np.zeros_like(dist)
        for i in range(dist.shape[0]):
            idx = np.argsort(dist[i])[:k + 1]  # include self
            S[i, idx] = 1
        S = (S + S.T) / 2
    else:
        raise ValueError("Unsupported similarity metric")

    np.fill_diagonal(S, 0)
    if normalized:
        S = normalize_graph(S)
    return S


def normalize_graph(S):
    """
    Normalize similarity matrix into symmetric stochastic matrix.
    """
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.sum(S, axis=1)) + 1e-8))
    return D_inv_sqrt @ S @ D_inv_sqrt


def compute_laplacian(S):
    """
    Compute Laplacian matrix from similarity matrix.
    """
    D = np.diag(np.sum(S, axis=1))
    return D - S


def preprocess_similarity_graph(X_views, metric='rbf', fuse_method='mean', **kwargs):
    """
    Construct fused similarity matrix and Laplacian from multi-view features.

    Parameters:
    -----------
    X_views : List[np.ndarray]
        List of feature matrices (d_v x n)
    metric : str
        Similarity computation method for each view
    fuse_method : str
        Fusion method: ['mean', 'max', 'adaptive']
    
    Returns:
    --------
    S_fused : np.ndarray
        Fused similarity matrix
    L_fused : np.ndarray
        Corresponding Laplacian matrix
    """
    S_list = [compute_similarity_matrix(Xv, metric=metric, **kwargs) for Xv in X_views]
    
    if fuse_method == 'mean':
        S_fused = sum(S_list) / len(S_list)
    elif fuse_method == 'max':
        S_fused = np.maximum.reduce(S_list)
    elif fuse_method == 'adaptive':
        weights = [np.linalg.norm(Sv, 'fro') for Sv in S_list]
        weights = np.array(weights) / np.sum(weights)
        S_fused = sum(w * S for w, S in zip(weights, S_list))
    else:
        raise ValueError("Unsupported fusion method")

    S_fused = (S_fused + S_fused.T) / 2
    np.fill_diagonal(S_fused, 0)
    L_fused = compute_laplacian(S_fused)
    return S_fused, L_fused
