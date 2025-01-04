import torch
from torch.linalg import svd
from torch import qr

torch.manual_seed(1337)  # Setting a seed for reproducibility

n = 2500
m = n
r = 5
alpha = 0.4
c = 1

import time

def wthresh(X, threshold_type, threshold):
    if threshold_type == 'h':
        return torch.sign(X) * torch.maximum(torch.abs(X) - threshold, torch.zeros_like(X))
    elif threshold_type == 's':
        return torch.sign(X) * torch.maximum(torch.abs(X) - threshold, torch.zeros_like(X)) * (torch.abs(X) > threshold)
    else:
        raise ValueError('Invalid threshold_type. Use "h" for hard thresholding or "s" for soft thresholding.')

def AccAltProj(D, r, para=None):
    m, n = D.shape
    norm_of_D = torch.norm(D, 'fro')

    # Default/Inputed parameters
    max_iter = 100
    tol = 1e-5
    beta = 1 / (2 * (m * n)**(1 / 4))
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

    err = -1 * torch.ones(max_iter)
    timer = -1 * torch.ones(max_iter)

    tic = time.time()

    zeta = beta_init * svd(D, )[0]

    S = wthresh(D, 'h', zeta)

    U, Sigma, V = svd(D - S)
    V = V.t()
    L = U @ torch.diag(Sigma) @ V.t()

    zeta = beta * Sigma[0]
    S = wthresh(D - L, 'h', zeta)

    init_timer = time.time() - tic
    init_err = torch.norm(D - L - S, 'fro') / norm_of_D
    print(f'Init: error: {init_err}, timer: {init_timer}')

    # Main Algorithm
    for t in range(max_iter):
        tic = time.time()

        # # Trim
        # if trimming:
        #     U, V = trim(U, Sigma[:r, :r], V, mu[0], mu[1])

        # Update L
        Z = D - S
        Q1, R1 = qr(Z.t() @ U - V @ (Z @ V).t())
        Q2, R2 = qr(Z @ V - U @ (U.t() @ Z @ V))
        M = torch.cat([U.t() @ Z @ V, R1.t(), torch.cat([R2, torch.zeros_like(R2)], dim=0)], dim=1)
        U_of_M, Sigma, V_of_M = svd(M)
        V_of_M = V_of_M.t()
        U = torch.cat([U, Q2], dim=1) @ U_of_M[:, :r]
        V = torch.cat([V, Q1], dim=1) @ V_of_M[:, :r]
        L = U @ torch.diag(Sigma[:r]) @ V.t()

        # Update S
        zeta = beta * (Sigma[r] + gamma**t * Sigma[0])
        S = wthresh(D - L, 'h', zeta)

        # Stop Condition
        err[t] = torch.norm(D - L - S, 'fro') / norm_of_D
        timer[t] = time.time() - tic

        if err[t] < tol:
            print(f'Total {t + 1} iteration, final error: {err[t]}, '
                  f'total time without init: {torch.sum(timer[timer > 0])}, '
                  f'with init: {torch.sum(timer[timer > 0]) + init_timer}')
            return L, S

        print(f'Iteration {t + 1}: error: {err[t]}, timer: {timer[t]}')

    print(f'Maximum iterations reached, final error: {err[-1]}')
    return L, S

# Generate a RPCA problem
A_generater = torch.randn(m, r)
B_generater = torch.randn(r, n)
L_true = torch.mm(A_generater, B_generater)
norm_of_L_true = torch.norm(L_true, 'fro')

S_supp_idx = torch.randperm(m * n)[:int(round(alpha * m * n))]
S_range = c * torch.mean(torch.abs(L_true))
S_temp = 2 * S_range * torch.rand(m, n) - S_range
S_true = torch.zeros((m, n))
S_true.view(-1)[S_supp_idx] = S_temp.view(-1)[S_supp_idx]
norm_of_S_true = torch.norm(S_true, 'fro')

D = L_true + S_true

L1, S1 = AccAltProj(D, r, {})
L1_err = torch.norm(L1 - L_true, 'fro') / torch.norm(L_true, 'fro')
