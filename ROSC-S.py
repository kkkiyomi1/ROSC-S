import numpy as np
import scipy
from scipy.linalg import solve_sylvester, orth
from numpy.fft import fftn, ifftn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy

def schatten_p_norm(Z, p):

    U, s, Vh = np.linalg.svd(Z)

    return np.power(np.sum(np.power(s, p)), 1 / p)

def update_Zv(Wv, Xv, Rv, Qv, Gv, C3v, Ev, C1v, mu, nv):

    Yv = np.random.randn(163, 163)

    M1v = Rv - Qv / mu
    M2v = Gv - C3v / mu
    M3v = Gv.T @ Xv - Ev + C1v / mu

    T1v = np.linalg.pinv(Wv.T @ Xv)
    T2v = np.eye(Xv.shape[1])
    T3v = M1v + (Wv.T @ Xv).T 
    T4v = np.linalg.inv(Yv.T @ Yv + np.eye(Yv.shape[0])) @ T3v
    
    Zv = solve_sylvester(T1v, T2v, T4v)

    return Zv

def update_R(Z, Q, mu, p):

    A = Z + Q / mu

    A_fft = fftn(A)

    threshold = 1 / (mu * p)
    R_fft = np.maximum(0, np.abs(A_fft) - threshold) * np.sign(A_fft)

    R = ifftn(R_fft)

    return R

def update_Gv_and_psi(Zv, C3v, psi, mu, nv, Gv):

    C3v = np.random.rand(*Zv.shape)

    Kv = Zv + C3v / mu

    Gv = np.maximum(Gv, 0)  

    for i in range(nv):
        psi[i] = mu * (1 - np.sum(Gv[i, :])) / (nv - 1)

    return Gv, psi

def update_Ev(Wv, Xv, Zv, C1v, lambda_1, mu):

    v = lambda_1 / mu

    M = Wv.T @ Xv - Wv.T @ Xv @ Zv + C1v / mu

    E_v = np.sign(M) * np.maximum(np.abs(M) - v, 0)
    
    return E_v

def update_Wv(Xv, U, L, lambda_2, lambda_3):

    n = Xv.shape[1]
    U = np.random.rand(163, 163)
    I_n = np.eye(n)
    C = Xv @ (I_n + (2 * lambda_2 / lambda_3) * L) @ Xv.T
    B = Xv @ U.T

    return GPI(C, B, Worig=None)

def GPI(A, B, Worig=None):
    m, k = B.shape
    alpha = max(abs(np.linalg.eigvals(A)))

    err = 1
    t = 1
    Wv = orth(np.random.randn(m, k))

    A_til = alpha * np.eye(m) - A
    obj = []

    while t < 1200:
        M = 2 * A_til @ Wv + 2 * B
        u, s, vh = np.linalg.svd(M)
        Wv = u[:, :k] @ vh
        obj.append(np.trace(Wv.T @ A @ Wv - 2 * Wv.T @ B))
        
        if t >= 2:
            err = abs(obj[-1] - obj[-2])
        
        t += 1
        if err < 1e-6:
            break

    return Wv

def update_P(L, c):

    eigenvalues, eigenvectors = np.linalg.eigh(L)

    idx = np.argsort(eigenvalues)[::-1]
    top_vectors = eigenvectors[:, idx[:c]]

    return top_vectors

def update_U(X_v, W_v, U, lambda3, lambda4, learning_rate):

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
    m = len(W)
    n = X[0].shape[1]  

    T = [W[v].T @ X[v] for v in range(m)]

    S_new = np.zeros_like(S)

    for i in range(n):
        e_i = np.zeros(n)
        for j in range(n):
            s_ij = 0
            for v in range(m):
                s_ij += (T[v][i] - T[v][j]).T @ (T[v][i] - T[v][j])
            e_i[j] = s_ij

        si_v = np.mean([S[v][i] for v in range(m)], axis=0)  # Mean of S(v) for each view v
        si = np.maximum(si_v - (lambda_2 / 2) * e_i, 0)

        si = si / np.sum(si)

        S_new[i] = si

    return S_new

def update_lagrangian_multipliers(C1v, C3v, Q, Wv, Xv, Zv, Ev, Gv, R, mu, rho, mu_0):

    C1v_new = C1v + mu * (Wv.T @ Xv - Wv.T @ Xv @ Zv - Ev)
    C3v_new = C3v + mu * (Zv - Gv)
    Q_new = Q + mu * (Zv - R)
    mu_new = min(rho * mu, mu_0)

    return C1v_new, C3v_new, Q_new, mu_new

def original_feature_regularizer(U):

    n, k = U.shape

    R_FF = 0

    for j in range(k):

        U_j = np.delete(U, j, axis=1)

        d_j = np.linalg.pinv(U_j).dot(U[:, j])

        U_j_dj = U_j.dot(d_j)

        R_FF += np.dot(U[:, j], U_j_dj)

    R_FF /= (n - 1)

    return R_FF

def calculate_sensitivity_specificity(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    true_negative, false_positive, false_negative, true_positive = cm.ravel()

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    return sensitivity, specificity

def run(X, Y, testX, testY,seed, mu,lambda_values,nv, rho, mu_0):
    # 训练数据 (X)、训练标签 (Y)、测试数据 (testX)、测试标签 (testY)
    n_views = len(X)
    d = np.ones((n_views, 1)) 
    Slsit = list()

    for i_view in range(n_views):
        St = scipy.spatial.distance.cdist(X[i_view].transpose(), X[i_view].transpose())
        St = St * St
        St = np.exp(-1 * St)
        St = St - np.diag(np.diag(St))
        Slsit.append(St)

        if i_view == 0:
            Ss = St.copy()
        else:
            if Ss.shape != St.shape:
                print(f"形状不匹配：Ss的形状是 {Ss.shape}，而St的形状是 {St.shape}")
            else:
                Ss += St
    Ss /= n_views
    Ss = Ss - np.diag(np.diag(Ss))
    S = Ss.copy()

    Sk = (S.transpose() + S) / 2
    D_ = np.diag(np.sum(Sk, axis=1))
    LS = D_ - Sk

    Zv = [np.zeros((nv, X[i].shape[1])) for i in range(len(X))]  
    Wv = [np.zeros_like(X[i]) for i in range(len(X))]
    R = np.random.rand(163, 163)
    Z = np.random.rand(163, 163)
    Gv = np.random.rand(264, 163)
    known_length = len(Gv)
    psi = [0] * known_length  
    P = np.zeros([Y.shape[0], 3])
        

    Xv = X.copy()
    v = len(X)
    w = v-1
    Sv = np.random.rand(163, 163)
    Sw = np.random.rand(163, 163)
    Ev = np.random.rand(163, 163)

    C1v = np.random.rand(163, 163)
    C3v = np.random.rand(264, 163)
    Q = np.random.rand(163, 163)


    U = np.random.rand(264, 264)
    lambda_1,lambda_2,lambda_3,lambda_4 = lambda_values
    err = list()
    err.append(1e9)

    total_loss = list()
    for ite in range(2):
        loss_Z = schatten_p_norm(Z, p=3) 
        loss_WXU, loss_E, original_feature_loss,loss_trace, loss_S = 0, 0, 0, 0, 0
        for v in range(len(Wv)):
            loss_WXU += (lambda_3 / 2) * np.linalg.norm(np.matmul(Xv[i_view], Wv[i_view].T) - U, 'fro')**2
            loss_E += lambda_1 * np.linalg.norm(Ev, 1)
            loss_S += lambda_4 * np.linalg.norm(S[i_view] - Sv, 'fro')**2
            original_feature_loss = lambda_4 * original_feature_regularizer(U)  
            for v in range(len(Wv)):
                WX = np.matmul(Wv[i_view].T, Xv[i_view])
                loss_trace += lambda_2 * np.trace(np.matmul(np.matmul(WX, LS), WX.T))

                
                total_loss = loss_Z + loss_WXU + loss_E + original_feature_loss + loss_trace +loss_S
                err.append(total_loss)
                if abs(err[ite + 1] - err[ite]) < 1e-3:
                    break

                # update Z^{(v)}
                Zv = update_Zv(Wv[i_view], Xv[i_view], R, Q, Gv, C3v, Ev, C1v, mu, nv)

                # update R
                R = update_R(Z, Q, mu, p=3)

                # update G^{(v)} , \psi^{(v)}
                Gv, psi = update_Gv_and_psi(Zv, C3v, psi, mu, nv, Gv)

                # update Ev
                Ev = update_Ev(Wv[i_view], Xv[i_view], Zv, C1v, lambda_1, mu)

                # update U
                U = update_U(Xv[i_view], Wv[i_view], U, lambda_3, lambda_4, learning_rate=0.01)

                # update Wv
                Wv[i_view] = update_Wv(Xv[i_view], U, LS, lambda_2, lambda_3)
                print("Wv[i_view] shape :", Wv[i_view].shape)

                # update S
                S = update_S(Wv[i_view], Xv[i_view], LS, S, lambda_2)

                # update P
                P = update_P(LS, c=nv)

                # update C1^{(v)}, C3^{(v)}, Q, psi^{(v)}, mu
                C1v, C3v, Q, mu = update_lagrangian_multipliers(C1v, C3v, Q, Wv[i_view], Xv[i_view], Zv, Ev, Gv, R, mu, rho, mu_0)


                
                Sk = (S.transpose() + S) / 2
                D_ = np.diag(np.sum(Sk, axis=0))
                LS = D_ - Sk


            for i_view in range(n_views):
                if i_view == 0:
                    Hall  = np.matmul(X[i_view],Wv[i_view].T)
                else:
                    Hall += np.matmul(X[i_view],Wv[i_view].T)
                
            param_grid = {
                'n_neighbors': range(1, 31),
                'metric': ['euclidean', 'manhattan', 'minkowski'],
                'weights': ['uniform', 'distance']
            }

            grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
            grid_search.fit(Hall, Y)

            knn = grid_search.best_estimator_

            knn.fit(Hall, Y)
            testHall = np.matmul(testX[i_view],Wv[i_view].T)
            y_pred = knn.predict(testHall)

            Acc = accuracy_score(testY, y_pred)
            Sen, Spe = calculate_sensitivity_specificity(testY, y_pred)
            Auc = roc_auc_score(testY, knn.predict_proba(testHall)[:, 1])  

            return Acc, Sen, Spe, Auc


            
        
        
