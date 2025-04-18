import numpy as np


def initialize_variables(X, Y, nv, strategy='orthogonal', seed=42):
    """
    Initialize model variables with various strategies.

    Parameters:
    -----------
    X : List[np.ndarray]
        Multi-view data (each of shape d_v x n)
    Y : np.ndarray
        One-hot or class-label matrix
    nv : int
        Latent dimensionality
    strategy : str
        Initialization strategy: ['random', 'zero', 'orthogonal']
    
    Returns:
    --------
    variables : dict
        Dictionary containing all model variables
    """
    np.random.seed(seed)
    n_views = len(X)
    n = X[0].shape[1]
    d_list = [Xv.shape[0] for Xv in X]
    label_dim = Y.shape[1] if Y.ndim > 1 else len(np.unique(Y))

    def init_matrix(shape):
        if strategy == 'zero':
            return np.zeros(shape)
        elif strategy == 'random':
            return np.random.randn(*shape)
        elif strategy == 'orthogonal':
            A = np.random.randn(*shape)
            return np.linalg.qr(A)[0] if shape[0] >= shape[1] else np.linalg.qr(A.T)[0].T
        else:
            raise ValueError("Invalid init strategy")

    variables = {
        'Zv': [init_matrix((nv, n)) for _ in range(n_views)],
        'Wv': [init_matrix((d_list[v], nv)) for v in range(n_views)],
        'R': init_matrix((nv, n)),
        'Z': init_matrix((nv, n)),
        'Gv': init_matrix((label_dim, nv)),
        'psi': [0] * label_dim,
        'P': np.zeros((label_dim, 3)),
        'Ev': init_matrix((nv, n)),
        'C1v': init_matrix((nv, n)),
        'C3v': init_matrix((label_dim, nv)),
        'Q': init_matrix((nv, n)),
        'U': init_matrix((label_dim, label_dim)),
        'S': np.random.rand(n, n),  # may be overwritten by preprocessing
    }

    return variables
