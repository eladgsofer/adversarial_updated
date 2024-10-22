import numpy as np
from scipy.linalg import svd, qr
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
import time


def AccAltProj(D, r, para=None):
    m, n = D.shape
    norm_of_D = np.linalg.norm(D, 'fro')

    # Default/Inputed parameters
    max_iter = 100
    tol = 1e-5
    beta = 1 / (2 * np.power(m * n, 1 / 4))
    beta_init = 4 * beta
    gamma = 0.7
    mu = 5
    trimming = False

    if para is not None:
        beta_init = para.get('beta_init', beta_init)
        beta = para.get('beta', beta)
        gamma = para.get('gamma', gamma)
        mu = para.get('mu', mu)
        # trimming = para.get('trimming', trimming)
        max_iter = para.get('max_iter', max_iter)
        tol = para.get('tol', tol)

    err = -1 * np.ones(max_iter)
    timer = -1 * np.ones(max_iter)

    tic = time.time()

    zeta = beta_init * svds(D, 1)[1][0]

    S = wthresh(D, 'h', zeta)

    U, Sigma, V = svds(D - S, r)
    V = V.T
    L = U @ np.diag(Sigma) @ V.T

    zeta = beta * Sigma[0]
    S = wthresh(D - L, 'h', zeta)

    init_timer = time.time() - tic
    init_err = np.linalg.norm(D - L - S, 'fro') / norm_of_D
    print(f'Init: error: {init_err}, timer: {init_timer}')

    # Main Algorithm
    for t in range(max_iter):
        tic = time.time()

        # # Trim
        # if trimming:
        #     U, V = trim(U, Sigma[:r, :r], V, mu[0], mu[1])

        # Update L
        Z = D - S
        Q1, R1 = qr(Z.T @ U - V @ (((Z @ V).T)@U), mode='economic')
        Q2, R2 = qr(Z @ V - U @ (U.T @ Z @ V), mode='economic')
        # Check how SVD works in MATLAB vs python
        M = np.block([[U.T @ Z @ V, R1.T], [R2, np.zeros_like(R2)]])
        U_of_M, Sigma, V_of_M = svd(M, full_matrices=False)
        V_of_M = V_of_M.T
        U = np.hstack([U, Q2]) @ U_of_M[:, :r]
        V = np.hstack([V, Q1]) @ V_of_M[:, :r]
        L = U @ np.diag(Sigma[:r]) @ V.T

        # Update S
        zeta = beta * (Sigma[r] + np.power(gamma, t) * Sigma[0])
        S = wthresh(D - L, 'h', zeta)

        # Stop Condition
        err[t] = np.linalg.norm(D - L - S, 'fro') / norm_of_D
        timer[t] = time.time() - tic

        if err[t] < tol:
            print(f'Total {t + 1} iteration, final error: {err[t]}, '
                  f'total time without init: {np.sum(timer[timer > 0])}, '
                  f'with init: {np.sum(timer[timer > 0]) + init_timer}')
            return L, S

        print(f'Iteration {t + 1}: error: {err[t]}, timer: {timer[t]}')

    print(f'Maximum iterations reached, final error: {err[-1]}')
    return L, S


# Helper function wthresh
def wthresh(X, threshold_type, threshold):
    if threshold_type == 'h':
        return np.sign(X) * np.maximum(np.abs(X) - threshold, 0)
    elif threshold_type == 's':
        return np.sign(X) * np.maximum(np.abs(X) - threshold, 0) * (np.abs(X) > threshold)
    else:
        raise ValueError('Invalid threshold_type. Use "h" for hard thresholding or "s" for soft thresholding.')
