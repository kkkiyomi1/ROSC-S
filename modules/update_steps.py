import numpy as np
from numpy.fft import fftn, ifftn
from scipy.linalg import solve_sylvester, orth


def schatten_p_norm(Z, p):
    """
    Compute the Schatten-p norm of a matrix Z.
    Reference: Section 3.2 of the paper - low-rank tensor regularization.
    """
    U, s, Vh = np.linalg.svd(Z)
    return np.power(np.sum(np.power(s, p)), 1 / p)


def update_Zv(Wv, Xv, Rv, Qv, Gv, C3v, Ev, C1v, mu, nv):
    """
    Update the representation matrix Z^{(v)} via Sylvester equation.
    """
    Yv = np.random.randn(nv, nv)
    T1v = np.linalg.pinv(Wv.T @ Xv)
    T2v = np.eye(Xv.shape[1])
    T3v = Rv - Qv / mu + Gv - C3v / mu + Wv.T @ Xv - Ev + C1v / mu
    T4v = np.linalg.inv(Yv.T @ Yv + np.eye(Yv.shape[0])) @ T3v
    Zv = solve_sylvester(T1v, T2v, T4v)
    return Zv


def update_R(Z, Q, mu, p):
    """
    Update tensor R using FFT-based soft-thresholding in frequency domain.
    """
    A = Z + Q / mu
    A_fft = fftn(A)
    threshold = 1 / (mu * p)
    R_fft = np.maximum(0, np.abs(A_fft) - threshold) * np.sign(A_fft)
    R = ifftn(R_fft).real
    return R


def update_Gv_and_psi(Zv, C3v, psi, mu, nv, Gv):
    """
    Update graph matrix G^{(v)} and Lagrangian ψ using residual adjustment.
    """
    Kv = Zv + C3v / mu
    Gv = np.maximum(Gv, 0)

    for i in range(nv):
        psi[i] = mu * (1 - np.sum(Gv[i, :])) / (nv - 1)
    return Gv, psi


def update_Ev(Wv, Xv, Zv, C1v, lambda_1, mu):
    """
    Sparse noise matrix E^{(v)} update via soft-thresholding.
    """
    v = lambda_1 / mu
    M = Wv.T @ Xv - Wv.T @ Xv @ Zv + C1v / mu
    E_v = np.sign(M) * np.maximum(np.abs(M) - v, 0)
    return E_v


def GPI(A, B, Worig=None, max_iter=1200, tol=1e-6):
    """
    Generalized Power Iteration method for solving orthogonal matrix projection.
    """
    m, k = B.shape
    alpha = max(abs(np.linalg.eigvals(A)))
    err = 1
    t = 1
    Wv = orth(np.random.randn(m, k))
    A_til = alpha * np.eye(m) - A
    obj = []

    while t < max_iter:
        M = 2 * A_til @ Wv + 2 * B
        u, s, vh = np.linalg.svd(M)
        Wv = u[:, :k] @ vh
        obj.append(np.trace(Wv.T @ A @ Wv - 2 * Wv.T @ B))

        if t >= 2:
            err = abs(obj[-1] - obj[-2])
        if err < tol:
            break
        t += 1
    return Wv


def update_Wv(Xv, U, L, lambda_2, lambda_3):
    """
    Update projection matrix W^{(v)} using GPI.
    """
    n = Xv.shape[1]
    I_n = np.eye(n)
    C = Xv @ (I_n + (2 * lambda_2 / lambda_3) * L) @ Xv.T
    B = Xv @ U.T
    return GPI(C, B)


def update_P(L, c):
    """
    Get top-c eigenvectors from Laplacian matrix for graph regularization.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)[::-1]
    top_vectors = eigenvectors[:, idx[:c]]
    return top_vectors


def update_U(X_v, W_v, U, lambda3, lambda4, learning_rate):
    """
    Update the consensus representation matrix U using regularized gradient descent.
    """
    n, k = U.shape
    U_tilde = np.zeros_like(U)

    for i in range(k):
        U_noi = np.delete(U, i, axis=1)
        U_noi_processed = np.copy(U_noi)
        for j in range(k - 1):
            if np.dot(U[:, i], U_noi[:, j]) < 0:
                U_noi_processed[:, j] = -U_noi[:, j]
        U_tilde[:, i] = np.sum(U_noi_processed, axis=1)

    gradient = lambda3 * (U - np.dot(X_v, W_v.T)) + 2 * lambda4 * U_tilde
    U_update = U - learning_rate * gradient
    return U_update


def update_S(W, X, LS, S, lambda_2):
    """
    Update similarity matrix S based on multiple projected views.
    """
    m = len(W)
    n = X[0].shape[1]
    T = [W[v].T @ X[v] for v in range(m)]
    S_new = np.zeros_like(S)

    for i in range(n):
        e_i = np.zeros(n)
        for j in range(n):
            s_ij = sum((T[v][i] - T[v][j]).T @ (T[v][i] - T[v][j]) for v in range(m))
            e_i[j] = s_ij

        si_v = np.mean([S[v][i] for v in range(m)], axis=0)
        si = np.maximum(si_v - (lambda_2 / 2) * e_i, 0)
        si = si / np.sum(si)
        S_new[i] = si
    return S_new


def update_lagrangian_multipliers(C1v, C3v, Q, Wv, Xv, Zv, Ev, Gv, R, mu, rho, mu_0):
    """
    Update Lagrangian multipliers and penalty mu for augmented Lagrangian.
    """
    C1v_new = C1v + mu * (Wv.T @ Xv - Wv.T @ Xv @ Zv - Ev)
    C3v_new = C3v + mu * (Zv - Gv)
    Q_new = Q + mu * (Zv - R)
    mu_new = min(rho * mu, mu_0)
    return C1v_new, C3v_new, Q_new, mu_new


def original_feature_regularizer(U):
    """
    Compute redundancy among feature vectors in latent space.
    """
    n, k = U.shape
    R_FF = 0
    for j in range(k):
        U_j = np.delete(U, j, axis=1)
        d_j = np.linalg.pinv(U_j).dot(U[:, j])
        U_j_dj = U_j.dot(d_j)
        R_FF += np.dot(U[:, j], U_j_dj)
    R_FF /= (n - 1)
    return R_FF
