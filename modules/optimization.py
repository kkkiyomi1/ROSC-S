import numpy as np
from modules.update_steps import (
    schatten_p_norm, update_Zv, update_R, update_Gv_and_psi, update_Ev,
    update_U, update_Wv, update_S, update_P, update_lagrangian_multipliers,
    original_feature_regularizer
)


def optimize_model(X, Y, testX, testY, S, LS, mu, lambda_values, nv, rho, mu_0, vars_dict,
                   max_iters=50, tol=1e-3, learning_rate=0.01, verbose=True):
    """
    Perform alternating optimization for ROSC-S.

    Parameters:
    -----------
    X, Y: list of np.ndarray
        Multi-view data and labels.
    S, LS: np.ndarray
        Initial similarity matrix and Laplacian matrix.
    vars_dict: dict
        Pre-initialized variable dictionary.
    lambda_values: tuple
        Tuple of (lambda_1, lambda_2, lambda_3, lambda_4)

    Returns:
    --------
    optimized_vars: dict
        Final optimized variables (Z, W, U, etc.)
    """
    # Unpack lambda
    lambda_1, lambda_2, lambda_3, lambda_4 = lambda_values

    # Unpack variables
    Zv, Wv, R, Z, Gv, psi = vars_dict['Zv'], vars_dict['Wv'], vars_dict['R'], vars_dict['Z'], vars_dict['Gv'], vars_dict['psi']
    P, Ev, C1v, C3v, Q, U = vars_dict['P'], vars_dict['Ev'], vars_dict['C1v'], vars_dict['C3v'], vars_dict['Q'], vars_dict['U']

    # Initialize loss tracking
    loss_history = []

    for ite in range(max_iters):
        total_loss = 0

        # ----- Section 1: Objective Loss Calculation -----
        loss_Z = schatten_p_norm(Z, p=3)
        loss_WXU = sum((lambda_3 / 2) * np.linalg.norm(np.matmul(X[v], Wv[v].T) - U, 'fro') ** 2 for v in range(len(Wv)))
        loss_E = lambda_1 * np.linalg.norm(Ev, 1)
        loss_trace = sum(
            lambda_2 * np.trace(np.matmul(np.matmul(Wv[v].T @ X[v], LS), (Wv[v].T @ X[v]).T))
            for v in range(len(Wv))
        )
        loss_S = lambda_4 * sum(np.linalg.norm(S - S[v], 'fro') ** 2 for v in range(len(X)))
        loss_Ureg = lambda_4 * original_feature_regularizer(U)

        total_loss = loss_Z + loss_WXU + loss_E + loss_trace + loss_S + loss_Ureg
        loss_history.append(total_loss)

        if verbose:
            print(f"[Iteration {ite+1}] Total Loss: {total_loss:.4f}")

        if ite > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            print(f"Converged at iteration {ite+1}")
            break

        # ----- Section 2: Variable Updates -----
        for v in range(len(X)):
            Zv[v] = update_Zv(Wv[v], X[v], R, Q, Gv, C3v, Ev, C1v, mu, nv)

        Z = sum(Zv) / len(Zv)
        R = update_R(Z, Q, mu, p=3)

        Gv, psi = update_Gv_and_psi(Zv[0], C3v, psi, mu, nv, Gv)  # Only first view for simplicity
        Ev = update_Ev(Wv[0], X[0], Zv[0], C1v, lambda_1, mu)
        U = update_U(X[0], Wv[0], U, lambda_3, lambda_4, learning_rate)

        for v in range(len(X)):
            Wv[v] = update_Wv(X[v], U, LS, lambda_2, lambda_3)

        S = update_S(Wv, X, LS, S, lambda_2)
        P = update_P(LS, c=nv)
        C1v, C3v, Q, mu = update_lagrangian_multipliers(C1v, C3v, Q, Wv[0], X[0], Zv[0], Ev, Gv, R, mu, rho, mu_0)

        # Recompute Laplacian
        Sk = (S.T + S) / 2
        D_ = np.diag(np.sum(Sk, axis=0))
        LS = D_ - Sk

    optimized_vars = {
        'Wv': Wv, 'U': U, 'Zv': Zv, 'Ev': Ev, 'S': S, 'Z': Z,
        'P': P, 'Gv': Gv, 'R': R, 'loss_history': loss_history
    }
    return optimized_vars
