import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn as nn
from scipy.linalg import eigvalsh
import numpy as np
import random
from utills import epoch, epoch_adversarial

import copy

from data import SimulatedData
import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.set_default_dtype(torch.float64)
BATCH_SIZE = 50


def create_data_set(H, n, m, k, N, batch_size=BATCH_SIZE, signal_dev=0.5, noise_dev=0.01):
    # Initialization
    x = torch.zeros(N, n)
    s = torch.zeros(N, m)
    #
    # s = np.zeros((1, m))
    # index_k = np.random.choice(m, k, replace=False)
    # s[:, index_k] = 0.5 * np.random.randn(k, 1).reshape([1, k])
    # s = torch.from_numpy(s).float()
    #
    # # x = Hs+w s.t w~N(0,1)
    # x = np.dot(H, s.T) + 0.01 * np.random.randn(n, 1)
    # x = torch.from_numpy(x).float()
    # return x.detach(), s.detach()

    # Create signals
    for i in range(N):
        # Create a k-sparsed signal s
        # x_sig, s_sig = generate_signal()
        # x[i, :], s[i, :] = x_sig.squeeze(1), s_sig.squeeze(0)
        index_k = np.random.choice(m, k, replace=False)
        peaks = 0.5 * np.random.randn(k, 1).reshape([1, k])

        s[i, index_k] = torch.from_numpy(peaks).to(s)

        # X = Hs+w
        # x[i, :] = H @ s[i, :] + noise_dev * torch.randn(n)
        x[i, :] = H @ s[i, :] + 0.01 * torch.randn(n)

    simulated = SimulatedData(x=x, H=H, s=s)
    data_loader = Data.DataLoader(dataset=simulated, batch_size=batch_size, shuffle=True)
    return data_loader


N = 100  # number of samples

# n = 150  # dim(x)
# m = 200  # dim(s)
# k = 4  # k-sparse signal

# Signal generation configuration
# m, n, k = 1500, 256, 5
n, m, k = 150, 200, 4

# Measurement matrix
H = torch.randn(n, m)
H /= torch.norm(H, dim=0)

import admm

admm.m = m
admm.N = N
admm.H = H
# Generate datasets
train_loader = create_data_set(H, n=n, m=m, k=k, N=N, batch_size=BATCH_SIZE)
test_loader = create_data_set(H, n=n, m=m, k=k, N=N, batch_size=N)
from admm import BIM, ADMM

def_attack_radius = 0.1
def_num_epochs = 3


def train(original_model, train_loader, valid_loader, num_epochs, attack_max_radius, save_models=False):
    """Train a network.
    Returns:
        loss_test {numpy} -- loss function values on test set
    """

    final_results_adv = {'admm': [], 'clean_model': [], 'robust_model': []}
    final_results_clean = {'admm': [], 'clean_model': [], 'robust_model': []}

    # adv_epsilon_vec = list(np.linspace(0.006, attack_max_radius, 4))
    adv_epsilon_vec = [0.006, 0.025, 0.038, 0.055, 0.069, attack_max_radius]

    for eps in adv_epsilon_vec:
        # Accumulate history for MSE vs epoch graph
        clean_model_adv, clean_model_clean, robust_model_adv, robust_model_clean = [], [], [], []
        for mode in ['clean_model', 'admm', 'robust_model']:
            # Initialization
            if mode in ['clean_model', 'robust_model']:
                model = copy.deepcopy(original_model)
                optimizer = torch.optim.SGD(model.parameters(), lr=5e-05, momentum=0.9, weight_decay=0)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

                for t in range(num_epochs):
                    model.train()
                    if mode == 'robust_model':
                        train_loss = epoch_adversarial(train_loader, model, BIM, opt=optimizer,
                                                       scheduler=scheduler, eps=eps)
                    else:
                        train_loss = epoch(train_loader, model, opt=optimizer, scheduler=scheduler)

                    # Testing phase - Test upon clean & adversarial test examples
                    model.eval()
                    clean_loss = epoch(valid_loader, model)
                    adv_loss = epoch_adversarial(valid_loader, model, BIM, eps=eps)
                    if mode == 'robust_model':
                        robust_model_adv.append(adv_loss)
                        robust_model_clean.append(clean_loss)
                    else:
                        clean_model_adv.append(adv_loss)
                        clean_model_clean.append(clean_loss)

                    print(*("{:.6f}".format(i) for i in (train_loss, clean_loss, adv_loss)), sep="\t")
            else:

                admm_model = ADMM.create_ADMM(H=H, max_iter=1000)
                adv_loss, clean_loss = 0., 0.
                for X, y in valid_loader:
                    yp, _ = admm_model(X)
                    clean_loss += F.mse_loss(yp.T, y, reduction="sum").item()
                clean_loss /= len(valid_loader.dataset)

                for X, y in valid_loader:
                    adv_x, _ = BIM(admm_model, X, y, eps=eps)
                    yp_adv, _ = admm_model(adv_x)
                    adv_loss += F.mse_loss(yp_adv.T, y, reduction="sum").item()

                adv_loss /= len(valid_loader.dataset)
            # Accumulate the last results for MSE vs epsilon graph
            final_results_adv[mode].append(adv_loss)
            final_results_clean[mode].append(clean_loss)

            if mode == 'robust_model':
                robust_model = copy.deepcopy(model)
            elif mode == 'clean_model':
                clean_model = copy.deepcopy(model)

            if save_models and mode != 'admm':
                torch.save(model.state_dict(), 'admm_{0}_{1}_epochs_{2}.npy'.format(mode, eps, num_epochs))

            print("mode {0} epsilon {1} ISTA adversarial loss: {2} clean loss {3}".format(mode, eps, adv_loss,
                                                                                          clean_loss))

        plot_loss_surface_trajectories(robust_model, clean_model, eps)

    plot_mse_vs_epsilon_graphs(adv_epsilon_vec, final_results_clean, final_results_adv)


def plot_mse_vs_epsilon_graphs(adv_epsilon_vec, final_results_clean, final_results_adv):
    plt.figure()
    plt.title('BIM max Epsilon {0}'.format(adv_epsilon_vec[-1]))
    plt.plot(adv_epsilon_vec, final_results_adv['robust_model'], label='admm-robust-model-adv-data', color='b', linewidth=1)
    plt.plot(adv_epsilon_vec, final_results_adv['clean_model'], label='admm-clean_model-adv-data', color='r', linewidth=1)
    plt.plot(adv_epsilon_vec, final_results_adv['admm'], label='admm-adv-data', color='g', linewidth=1)
    plt.xlabel('epsilon')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('BIM max Epsilon {0}'.format(adv_epsilon_vec[-1]))
    plt.plot(adv_epsilon_vec, final_results_clean['robust_model'], label='admm-robust-model-clean-data', color='b',
             linewidth=1)
    plt.plot(adv_epsilon_vec, final_results_clean['clean_model'], label='admm-clean_model-clean-data', color='r',
             linewidth=1)
    plt.plot(adv_epsilon_vec, final_results_clean['admm'], label='admm-clean-data', color='g', linewidth=1)
    plt.xlabel('epsilon')

    plt.ylabel('MSE')
    plt.legend()
    plt.show()


def plot_loss_surface_trajectories(robust_model, clean_model, epsilon):
    # Plotting trajectories upon ISTA_gt loss surface
    x, s = train_loader.dataset[0][0].unsqueeze(0).T, train_loader.dataset[0][1].unsqueeze(0)
    signals = [(x.double(), s.double())]
    kwargs = admm.execute(signals=signals, H_mat=H, plot_graphs=False, get_exec_params_mode=True)
    robust_model.eval()
    clean_model.eval()

    # TODO - why A, B is using N-number of samples as dimension?
    _, adv_s_hats_traj = robust_model.forward(x.T, acc_s_hat=True)
    _, clean_s_hats_traj = clean_model.forward(x.T, acc_s_hat=True)
    kwargs['clean_trajectory'] = clean_s_hats_traj
    kwargs['adv_trajectory'] = adv_s_hats_traj
    kwargs['steps'] = 800
    kwargs['distance'] = 3
    Z_gt, Z_adv, traj_clean, traj_adv = kwargs['gt_model'].random_plane(**kwargs)
    plt.figure()
    # Plot surface
    cs = plt.contour(Z_gt)

    # Plot trajectories
    clean_x_cor = [t['i'] for t in traj_clean]
    clean_y_cor = [t['j'] for t in traj_clean]
    plt.scatter(clean_x_cor, clean_y_cor, color='red', marker='o', zorder=5)
    adv_x_cor = [t['i'] for t in traj_adv]
    adv_y_cor = [t['j'] for t in traj_adv]
    plt.scatter(adv_x_cor, adv_y_cor, color='blue', marker='x', zorder=6)

    # Styling
    plt.clabel(cs, inline=1, fontsize=10)
    plt.colorbar(cs)
    plt.xlabel(r'$u_2$')
    plt.ylabel(r'$u_1$')
    # plt.style.use('plot_style.txt')
    plt.title("ADMM loss surface with trajectories, epsilon={0}".format(epsilon))
    # plt.savefig("ISTA_2D_LOSS_GT.pdf", bbox_inches='tight')
    plt.legend(['LADMM-clean trajectory', 'LISTA-adv trajectory'])
    plt.show()


class LADMM_Model(nn.Module):
    def __init__(self, n, m, T=6, rho=0.01, H=H, lambda_=12.5, mu=0.00005):
        super(LADMM_Model, self).__init__()
        self.n, self.m = n, m
        self.H = H

        # ISTA Iterations
        self.T = T

        # Initialization
        self.rho = nn.Parameter(torch.ones(T + 1, 1, 1) * rho, requires_grad=True)  # Lagrangian Multiplier
        self.lambda_ = nn.Parameter(torch.ones(T + 1, 1, 1) * lambda_, requires_grad=True)
        self.mu = nn.Parameter(torch.ones(T + 1, 1, 1) * mu, requires_grad=True)

    def _shrink(self, s, beta, rho):
        return beta * F.softshrink(s / beta, lambd=rho)

    def forward(self, x, acc_s_hat=False):
        """

        Args:
            x: a sparse signal observation
        Returns: S reconstruction
        """
        # H.shape[1] = 200, x.shape[0[ = 512 (batch size)
        # s_prev = torch.zeros(x.shape[0], self.H.shape[1])
        u_prev = torch.zeros((x.shape[0], self.H.shape[1]))
        v_prev = torch.zeros((x.shape[0], self.H.shape[1]))

        #################### Iteration 0 ####################

        left_term = torch.linalg.inv(self.H.T @ self.H + self.rho[0, :, :] * torch.eye(self.H.shape[1]))

        right_term = (self.H.T @ x.T).T + self.rho[0, :, :] * (v_prev - u_prev)

        s = (left_term @ right_term.T).T
        s_hat_ls = [s.detach()]
        v = self._shrink(s + u_prev,
                         self.rho[0, :, :] / (2 * self.lambda_[0, :, :]),
                         rho=self.rho[0, :, :].item())
        u = u_prev + self.mu[0, :, :] * (s - v)

        #################### Iteration 1<=i<=K ####################

        for i in range(1, self.T + 1):
            s_prev, v_prev, u_prev = s, v, u

            # left_term = (H^TH+rho*I)^-1
            left_term = torch.linalg.inv(self.H.T @ self.H + self.rho[i, :, :] * torch.eye(self.H.shape[1]))

            right_term = (self.H.T @ x.T).T + self.rho[i, :, :] * (v_prev - u_prev)

            # Update s_k+1 = ((H^T)H+2λI)^−1(H^T x+2λ(vk−uk)).
            s = (left_term @ right_term.T).T
            if acc_s_hat:
                s_hat_ls.append(s.detach())

            # Update vk+1 = prox_(1/2λϕ)(sk+1 + uk)
            v = self._shrink(s + u_prev, self.rho[i, :, :] / (2 * self.lambda_[i, :, :]), rho=self.rho[i, :, :].item())

            # Update uk+1 = uk + μ (sk+1 − vk+1).
            u = u_prev + self.mu[i, :, :] * (s - v)

        return s, s_hat_ls

    @classmethod
    def create_ladmm_model(cls, H, T):
        # Is there randomness at creating each time new instance?
        n = H.shape[0]
        m = H.shape[1]
        return cls(n=n, m=m, T=T)


def ladmm_apply(train_loader, test_loader, T, H):
    ladmm = LADMM_Model.create_ladmm_model(H, T)
    train(ladmm, train_loader, test_loader, num_epochs=def_num_epochs,
          attack_max_radius=def_attack_radius, save_models=True)


def start():
    T_ADMM = 20
    # Train and apply LISTA with T iterations / layers
    ladmm_apply(train_loader, test_loader, T_ADMM, H)


if __name__ == '__main__':
    start()

    # 1. MLSP
    # 2. RPCA attack
    # 3. Loss plots Robust LISTA vs LISTA vs ISTA
    # 4. Trajectory of LISTA/ LISTA_ADV upon ISTA loss surface

    # JL lemma - Johnson Lindenshtauch Lemma- norm0 vs norm 1

    # Get these values from ISTA
    # (train_loader.dataset[0][0].float().unsqueeze(0), train_loader.dataset[0][1].float().unsqueeze(0))

    # clean_model = LISTA_Model.create_lista_model()
    # clean_model.load_state_dict(torch.load('{0}_{1}_epochs_{2}.npy'.format('clean_model', '0.025', '10')))
    #
    # robust_model = LISTA_Model.create_lista_model()
    # clean_model.load_state_dict(torch.load('{0}_{1}_epochs_{2}.npy'.format('robust_model', '0.025', '10')))

    # Plot MSE vs epsilon graphs

    # ISTA_Z_gt = np.load('/Users/elad.sofer/src/ADVERSARIAL_SENSITIVTY/data/matrices/loss_surfaces/ISTA_Z_gt.npy')
    # ISTA_x = kwargs['x'].detach().T.double()
    # ISTA_x.require_grad = False
    # ISTA_s = np.load('/Users/elad.sofer/src/ADVERSARIAL_SENSITIVTY/data/matrices/loss_surfaces/ISTA_S.npy')
    # from copy import deepcopy
    # clean_model = deepcopy(model)
    # robust_model = deepcopy(model)
    # clean_model.load_state_dict(torch.load('clean_model_{0}_epochs_{1}.npy'.format(eps, num_epochs)))
    # robust_model.load_state_dict(torch.load('robust_model_{0}_epochs_{1}.npy'.format(eps, num_epochs)))

    # np.save(adv_s_hats_traj, 'adv_s_hats_traj.npy')
    # np.save(clean_s_hats_traj, 'clean_s_hats_traj.npy')

    # def random_plane(self, gt_model, adv_model, x, adv_x, distance=3, steps=20,
    #                  deepcopy_model=False, dir_one=None, dir_two=None, LISTA_clean_trajectory=None,
    #                  LISTA_adv_trajectory=None) -> np.ndarray:

    # todo - bug - 0'z signals

    # TODO Loss surface - LISTA - NOT-PERTUBED LISTA-
    # torch.save(model.state_dict(), "model_robust_{0}.pt".format(eps))

# def ista_apply(test_loader, T, H, rho=0.5):
#     H = H.cpu()
#     m = H.shape[1]
#     L = float(eigvalsh(H.t() @ H, eigvals=(m - 1, m - 1)))
#
#     # Aggregate T iterations' MSE loss
#     losses = np.zeros((len(test_loader.dataset)))
#     loss = []
#
#     for idx, (x, b_s) in enumerate(test_loader.dataset):
#         loss.append(ista(x=x, H=H, b_s=b_s, rho=rho, L=L, max_itr=T))
#
#     loss = np.array(loss)
#
#     return loss.mean()


# def ista(x, H, b_s, rho=0.5, L=1, max_itr=300, threshold=1e-3):
#     # loss_vs_iter = np.zeros(max_itr)
#     s_hat = torch.zeros(H.shape[1])
#     proj = torch.nn.Softshrink(lambd=rho / L)
#     s_prev = s_hat - 1 / L * (H.T @ (H @ s_hat - x)) + 2 * threshold
#     for idx in range(max_itr):
#         s_tild = s_hat - 1 / L * (H.T @ (H @ s_hat - x))
#         s_hat = proj(s_tild)
#         # Aggregate each iteration's MSE loss
#         loss = F.mse_loss(s_hat, b_s, reduction="sum").data.item()
#         if torch.sum(torch.abs(s_hat - s_prev)).item() <= threshold:
#             break
#         s_prev = s_hat
#
#     return loss


# # Signal generation configuration
# m, n, k = 1500, 256, 5
# Psi = np.eye(m)
# Phi = np.random.randn(n, m)
# Phi = np.transpose(orth(np.transpose(Phi)))
# H = Phi
# H = torch.from_numpy(H).float()
#
#
# def generate_signal():
#     """
#     Generate a sparse signal 's' and its observation 'x' using the model x = Hs + w, where w is a Gaussian noise.
#     """
#     s = np.zeros((1, m))
#     index_k = np.random.choice(m, k, replace=False)
#     s[:, index_k] = 0.5 * np.random.randn(k, 1).reshape([1, k])
#     s = torch.from_numpy(s).float()
#
#     # x = Hs+w s.t w~N(0,1)
#     x = np.dot(H, s.T) + 0.01 * np.random.randn(n, 1)
#     x = torch.from_numpy(x).float()
#     return x.detach(), s.detach()


# x_exm, s_exm = test_loader.dataset.__getitem__(5)
# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(x_exm, label='observation')
# plt.xlabel('Index', fontsize=10)
# plt.ylabel('Value', fontsize=10)
# plt.legend()
# plt.subplot(2, 1, 2)
# plt.plot(s_exm, label='sparse signal', color='k')
# plt.xlabel('Index', fontsize=10)
# plt.ylabel('Value', fontsize=10)
# plt.legend()
# plt.show()
