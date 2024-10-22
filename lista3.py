for mode in ['clean_model', 'robust_model']:
    for lista_mode in ['solver', 'all']:

        # for mode in ['ista', 'clean_model', 'robust_model']:
        # # Initialization
        if mode in ['clean_model', 'robust_model']:

            model = copy.deepcopy(original_model)
            if lista_mode == 'solver':
                model.B.requires_grad = False
                model.A.requires_grad = False
            else:
                model.B.requires_grad = True
                model.A.requires_grad = True

import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn as nn
from scipy.linalg import eigvalsh
import numpy as np
import random
from utills import generate_signal

from ista import BIM
import copy

from data import SimulatedData
import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.set_default_dtype(torch.float64)
BATCH_SIZE = 50


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

def create_data_set(H, n, m, k, N=1000, batch_size=BATCH_SIZE, signal_dev=0.5, noise_dev=0.01):
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


N = 1200  # number of samples

# n = 150  # dim(x)
# m = 200  # dim(s)
# k = 4  # k-sparse signal

# Signal generation configuration
m, n, k = 1500, 256, 5

# Measurement matrix
H = torch.randn(n, m)
H /= torch.norm(H, dim=0)

import ista

ista.m = m
ista.N = N
ista.H = H
# Generate datasets
train_loader = create_data_set(H, n=n, m=m, k=k, N=N, batch_size=BATCH_SIZE)
test_loader = create_data_set(H, n=n, m=m, k=k, N=N, batch_size=N)


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


def epoch(loader, model, opt=None, scheduler=None):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0., 0.
    for X, y in loader:
        # X, y = X.to(device), y.to(device)
        yp, _ = model(X)
        loss = F.mse_loss(yp, y, reduction="sum")
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_loss += loss.data.item()

    if scheduler:
        scheduler.step()
    return total_loss / len(loader.dataset)


def epoch_adversarial(loader, model, attack, opt=None, scheduler=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0., 0.
    for X, y in loader:
        # X, y = X.to(device), y.to(device)
        adv_x, _ = attack(model, X, y, **kwargs)  # def BIM(model, x, s_gt, eps=0.1, alpha=0.01, steps=5):
        yp, e_loss = model(adv_x)
        loss = F.mse_loss(yp, y, reduction="sum")
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_loss += loss.data.item()
    if scheduler:
        scheduler.step()
    return total_loss / len(loader.dataset)


def train(original_model, train_loader, valid_loader, num_epochs=10, attack_max_radius=0.1, save_models=False):
    """Train a network.
    Returns:
        loss_test {numpy} -- loss function values on test set
    """
    final_results_adv = {'ista': [], 'clean_model': [], 'robust_model': []}
    final_results_clean = {'ista': [], 'clean_model': [], 'robust_model': []}
    # TODO - Make A,B parameters in the LISTA.
    adv_epsilon_vec = list(np.linspace(0.007, attack_max_radius, 4))
    # for eps in adv_epsilon_vec:


    for eps in [0.025]:
        clean_model_adv, clean_model_clean, robust_model_adv, robust_model_clean = [], [], [], []
        for mode in ['clean_model', 'robust_model']:
            # for mode in ['ista', 'clean_model', 'robust_model']:
            # # Initialization
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

                ista_model = ISTA.create_ISTA(H=H, max_iter=1000)
                adv_loss, clean_loss = 0., 0.
                for X, y in valid_loader:
                    yp, _ = ista_model(X)
                    clean_loss += F.mse_loss(yp.T, y, reduction="sum")
                clean_loss /= len(valid_loader.dataset)

                for X, y in valid_loader:
                    adv_x, _ = BIM(ista_model, X, y, eps=eps)
                    yp_adv, _ = ista_model(adv_x)
                    adv_loss += F.mse_loss(yp_adv.T, y, reduction="sum")

                adv_loss /= len(valid_loader.dataset)

            final_results_adv[mode].append(adv_loss)
            final_results_clean[mode].append(clean_loss)
            if mode == 'robust_model':
                robust_model = copy.deepcopy(model)
            elif mode == 'clean_model':
                clean_model = copy.deepcopy(model)

            if save_models:
                torch.save(model.state_dict(), '{0}_{1}_epochs_{2}.npy'.format(mode, eps, num_epochs))

            print("mode {0} epsilon {1} ISTA adversarial loss: {2} clean loss {3}".format(mode, eps, adv_loss,
                                                                                          clean_loss))

    # Get these values from ISTA
    #(train_loader.dataset[0][0].float().unsqueeze(0), train_loader.dataset[0][1].float().unsqueeze(0))

    # clean_model = LISTA_Model.create_lista_model()
    # clean_model.load_state_dict(torch.load('{0}_{1}_epochs_{2}.npy'.format('clean_model', '0.025', '10')))
    #
    # robust_model = LISTA_Model.create_lista_model()
    # clean_model.load_state_dict(torch.load('{0}_{1}_epochs_{2}.npy'.format('robust_model', '0.025', '10')))

    x, s = train_loader.dataset[0][0].unsqueeze(0).T, train_loader.dataset[0][1].unsqueeze(0)
    kwargs = ista.execute(signals= [(x.double(), s.double())], H_mat=H)




    # Plotting trajectories upon ISTA_gt loss surface
    # ISTA_Z_gt = np.load('/Users/elad.sofer/src/ADVERSARIAL_SENSITIVTY/data/matrices/loss_surfaces/ISTA_Z_gt.npy')

    # ISTA_x = kwargs['x'].detach().T.double()
    # ISTA_x.require_grad = False


    # ISTA_s = np.load('/Users/elad.sofer/src/ADVERSARIAL_SENSITIVTY/data/matrices/loss_surfaces/ISTA_S.npy')

    # from copy import deepcopy
    # clean_model = deepcopy(model)
    # robust_model = deepcopy(model)
    # clean_model.load_state_dict(torch.load('clean_model_{0}_epochs_{1}.npy'.format(eps, num_epochs)))
    # robust_model.load_state_dict(torch.load('robust_model_{0}_epochs_{1}.npy'.format(eps, num_epochs)))
    robust_model.eval()
    clean_model.eval()

    # TODO - why A, B is using N-number of samples as dimension?
    _, adv_s_hats_traj = robust_model.forward(x.T, acc_s_hat=True)
    _, clean_s_hats_traj = clean_model.forward(x.T, acc_s_hat=True)
    # np.save(adv_s_hats_traj, 'adv_s_hats_traj.npy')
    # np.save(clean_s_hats_traj, 'clean_s_hats_traj.npy')

    # def random_plane(self, gt_model, adv_model, x, adv_x, distance=3, steps=20,
    #                  deepcopy_model=False, dir_one=None, dir_two=None, LISTA_clean_trajectory=None,
    #                  LISTA_adv_trajectory=None) -> np.ndarray:

    # todo - bug - 0'z signals
    kwargs['LISTA_clean_trajectory'] = clean_s_hats_traj
    kwargs['LISTA_adv_trajectory'] = adv_s_hats_traj
    kwargs['steps'] = 8000
    kwargs['distance'] = 1.5

    Z_gt, Z_adv, traj_clean, traj_adv = kwargs['gt_model'].random_plane(**kwargs)

    # Plot Loss surface
    plt.figure()
    # Plot surface
    cs = plt.contour(Z_gt)

    # Plot trajectories
    clean_x_cor = [t['i'] for t in traj_clean]
    clean_y_cor = [t['j'] for t in traj_clean]
    plt.scatter(clean_x_cor, clean_y_cor, color='red', marker='o', zorder=5)
    adv_x_cor = [t['i'] for t in traj_adv]
    adv_y_cor = [t['j'] for t in traj_adv]
    plt.scatter(adv_x_cor, adv_y_cor, color='blue',marker='x', zorder=6)

    # Styling
    plt.clabel(cs, inline=1, fontsize=10)
    plt.colorbar(cs)
    plt.xlabel(r'$u_2$')
    plt.ylabel(r'$u_1$')
    # plt.style.use('plot_style.txt')
    plt.title("Loss surface of L_truth(s) = 0.5*||x-Hs|| + rho*||s| s.t (rho=0.01), epsilon=0.1")
    # plt.savefig("ISTA_2D_LOSS_GT.pdf", bbox_inches='tight')
    plt.show()
    pass
    # Plot Trajectories

    # plt.figure()
    # plt.title('BIM Epsilon {0}'.format(eps))
    # plt.plot(adv_epsilon_vec, final_results_adv['robust_model'], label='robust-model-adv-data', color='b', linewidth=1)
    # plt.plot(adv_epsilon_vec, final_results_adv['clean_model'], label='clean_model-adv-data', color='r', linewidth=1)
    # plt.plot(adv_epsilon_vec, final_results_adv['ista'], label='ista-adv-data', color='g',linewidth=1)
    # plt.xlabel('epsilon', fontsize=10)
    # plt.ylabel('MSE', fontsize=10)
    # plt.legend()
    # plt.show()
    #
    # plt.figure()
    # plt.title('BIM Epsilon {0}'.format(eps))
    # plt.plot(adv_epsilon_vec, final_results_clean['robust_model'], label='robust-model-clean-data', color='b', linewidth=1)
    # plt.plot(adv_epsilon_vec, final_results_clean['clean_model'], label='clean_model-clean-data', color='r', linewidth=1)
    # plt.plot(adv_epsilon_vec, final_results_clean['ista'], label='ista-clean-data', color='g',linewidth=1)
    # plt.xlabel('epsilon', fontsize=10)
    # plt.ylabel('MSE', fontsize=10)
    # plt.legend()
    # plt.show()
    # TODO Loss surface - LISTA - NOT-PERTUBED LISTA-
    # torch.save(model.state_dict(), "model_robust_{0}.pt".format(eps))


class LISTA_Model(nn.Module):
    T_LISTA = 10

    def __init__(self, n, m, T=6, rho=1.0, H=None):
        super(LISTA_Model, self).__init__()
        self.n, self.m = n, m
        self.H = H
        self.T = T  # ISTA Iterations
        self.rho = rho  # Lagrangian Multiplier
        self.A = nn.Linear(n, m, bias=False)  # Weight Matrix
        self.B = nn.Linear(m, m, bias=False)  # Weight Matrix
        # ISTA Stepsizes eta
        self.beta = nn.Parameter(torch.ones(T + 1, 1, 1), requires_grad=True)
        self.mu = nn.Parameter(torch.ones(T + 1, 1, 1), requires_grad=True)
        # Initialization
        if H is not None:
            self.A.weight.data = H.t()
            self.B.weight.data = H.t() @ H

    def _shrink(self, s, beta):
        return beta * F.softshrink(s / beta, lambd=self.rho)

    def forward(self, x, acc_s_hat=False):

        s_hat_ls = []

        s_hat = self._shrink(self.mu[0, :, :] * self.A(x), self.beta[0, :, :])
        for i in range(1, self.T + 1):
            s_hat = self._shrink(s_hat - self.mu[i, :, :] * self.B(s_hat) + self.mu[i, :, :] * self.A(x),
                                 self.beta[i, :, :], )
            # Aggregate each iteration's MSE loss
            if acc_s_hat:
                s_hat_ls.append(s_hat.detach())
        return s_hat, s_hat_ls

    @classmethod
    def create_lista_model(cls, H=H):
        # Is there randomness at creating each time new instance?
        n = H.shape[0]
        m = H.shape[1]
        return cls(n=n, m=m, T=cls.T_LISTA, H=H)


def lista_apply(train_loader, test_loader, T, H):
    lista = LISTA_Model.create_lista_model()
    train(lista, train_loader, test_loader, save_models=True)
    # validate(lista)


from ista import ISTA

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


# number of unfolded iteartions

T_LISTA = LISTA_Model.T_LISTA = 10
# T_ISTA = 20 * T_LISTA

# TODO play with the batch_size to reduce/increase adversarial examples. determine 1000=dataset.
# Train and apply LISTA with T iterations / layers
# ista_apply(test_loader, )
lista_mse_vs_iter = lista_apply(train_loader, test_loader, T_LISTA, H)
pass

# 1. MLSP
# 2. RPCA attack
# 3. Loss plots Robust LISTA vs LISTA vs ISTA
# 4. Trajectory of LISTA/ LISTA_ADV upon ISTA loss surface

# JL lemma - Johnson Lindenshtauch Lemma- norm0 vs norm 1
