import numpy as np
import AccAltProj

np.random.seed(1337)  # Setting a seed for reproducibility

n = 2500
m = n
r = 5
alpha = 0.4
c = 1

print(np.random.rand(1,3))
# Generate a RPCA problem
A_generater = np.random.randn(m, r)
B_generater = np.random.randn(r, n)
L_true = np.dot(A_generater, B_generater)
norm_of_L_true = np.linalg.norm(L_true, 'fro')

S_supp_idx = np.random.choice(m * n, size=int(round(alpha * m * n)), replace=False)
S_range = c * np.mean(np.abs(L_true))
S_temp = 2 * S_range * np.random.rand(m, n) - S_range
S_true = np.zeros((m, n))
S_true.flat[S_supp_idx] = S_temp.flat[S_supp_idx]
norm_of_S_true = np.linalg.norm(S_true, 'fro')

D = L_true + S_true

L1, S1 = AccAltProj.AccAltProj( D, r, {})
L1_err = np.linalg.norm(L1-L_true, 'fro') / np.linalg.norm(L_true,'fro')