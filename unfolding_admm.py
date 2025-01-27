import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn as nn
from scipy.linalg import eigvalsh
from utills import save_fig
import numpy as np
import random
from utills import epoch, epoch_adversarial, MODEL_PATH_TEMPLATE, save_object, plot_defense_graph, get_attack_func
from data_utils import create_data_set

import copy

import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.set_default_dtype(torch.float64)
BATCH_SIZE = 50
T_ADMM = 5

N = 1500  # number of samples

# n = 150  # dim(x)
# m = 200  # dim(s)
# k = 4  # k-sparse signal

# Signal generation configuration
# m, n, k = 1500, 256, 5
# n, m, k = 250, 500, 4
m, n, k = 1200, 256, 5

# Measurement matrix
H = torch.randn(n, m)
H /= torch.norm(H, dim=0)

import classic_admm

classic_admm.m = m
classic_admm.N = N
classic_admm.H = H
# Generate datasets
train_loader = create_data_set(H, n=n, m=m, k=k, N=N, batch_size=BATCH_SIZE)
test_loader = create_data_set(H, n=n, m=m, k=k, N=N, batch_size=N//4)

def_attack_radius = 0.1
def_num_epochs = 40


def inference(valid_loader, save_figures=False):
    """Train a network.
    Returns:
        loss_test {numpy} -- loss function values on test set
    """

    final_results_adv = {'admm': [], 'clean_model': [], 'robust_model': []}
    final_results_clean = {'admm': [], 'clean_model': [], 'robust_model': []}

    # adv_epsilon_vec = list(np.linspace(0.006, attack_max_radius, 4))
    # adv_epsilon_vec = [0.006, 0.025, 0.038, 0.055, 0.069, attack_max_radius]
    adv_epsilon_vec = [0.005, 0.025, 0.045, 0.065, 0.085]

    for eps in adv_epsilon_vec:
        # Accumulate history for MSE vs epoch graph

        for mode in ['clean_model', 'admm', 'robust_model']:
            # Initialization
            if mode in ['clean_model', 'robust_model']:
                path = MODEL_PATH_TEMPLATE.format(model='ladmm', attack='BIM',N=N,
                                                  mode=mode, epsilon=eps, epochs=def_num_epochs,
                                                  MBDL=str(True), K=T_ADMM)
                model = load_model_eval_model(path)
                clean_loss = epoch(valid_loader, model)
                adv_loss = epoch_adversarial(valid_loader, model, classic_admm.BIM, eps=eps)

            else:
                admm_model = classic_admm.ADMM.create_ADMM(H=H, max_iter=1000)
                adv_loss, clean_loss = 0., 0.
                for X, y in valid_loader:
                    yp, _ = admm_model(X)
                    clean_loss += F.mse_loss(yp.T, y, reduction="sum").item()
                clean_loss /= len(valid_loader.dataset)

                for X, y in valid_loader:
                    adv_x, _ = classic_admm.BIM(admm_model, X, y, eps=eps)
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

            print("mode {0} epsilon {1} ADMM adversarial loss: {2} clean loss {3}".format(mode, eps, adv_loss,
                                                                                          clean_loss))

        plot_loss_surface_trajectories(robust_model, clean_model, eps, save_figures)

    plot_object = {'adv_epsilon_vec': adv_epsilon_vec, 'final_results_clean': final_results_clean,
                   'final_results_adv': final_results_adv}
    object_fname = f'LADMM_defense_eps_{str(adv_epsilon_vec)}_"BIM".pkl'
    save_object(plot_object, object_fname)
    plot_defense_graph(adv_epsilon_vec, object_fname, "LADMM")


def train(original_model, train_loader, valid_loader, num_epochs, attack, attack_magnitudes, save_models=False,
          save_figures=True):
    """Train a network.
    Returns:
        loss_test {numpy} -- loss function values on test set
    """

    final_results_adv = {'admm': [], 'clean_model': [], 'robust_model': []}
    final_results_clean = {'admm': [], 'clean_model': [], 'robust_model': []}
    final_results_l_inf = {'ista': [], 'clean_model': [], 'robust_model': []}

    # adv_epsilon_vec = list(np.linspace(0.006, attack_max_radius, 4))
    # adv_epsilon_vec = [0.006, 0.025, 0.038, 0.055, 0.069, attack_max_radius]


    for eps in attack_magnitudes:
        if attack in ["BIM", "NIFGSM"]:
            attack_kwargs = dict(eps=eps)
        elif attack == "CW":
            attack_kwargs = dict(c=eps)
        else:
            raise Exception("Not implemented attack")

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
                        train_loss, _ = epoch_adversarial(train_loader, model, attack, opt=optimizer,
                                                       scheduler=scheduler, **attack_kwargs)
                    else:
                        train_loss = epoch(train_loader, model, opt=optimizer, scheduler=scheduler)

                    # Testing phase - Test upon clean & adversarial test examples
                    model.eval()
                    clean_loss = epoch(valid_loader, model)
                    if attack=="CW":
                        adv_loss, l_inf = epoch_adversarial(valid_loader, model, attack, **attack_kwargs)
                    else:
                        adv_loss, _ = epoch_adversarial(valid_loader, model, attack, **attack_kwargs)

                    print(*("{:.6f}".format(i) for i in (train_loss, clean_loss, adv_loss)), sep="\t")
            else:

                admm_model = classic_admm.ADMM.create_ADMM(H=H, max_iter=1000)
                adv_loss, clean_loss = 0., 0.
                L_inf = 0.
                for X, y in valid_loader:
                    yp, _ = admm_model(X)
                    clean_loss += F.mse_loss(yp.T, y, reduction="sum").item()
                clean_loss /= len(valid_loader.dataset)

                for X, y in valid_loader:
                    adv_x, _ = get_attack_func(attack_name=attack)(admm_model, X, y, **attack_kwargs)
                    if attack=="CW":
                        L_inf += abs(adv_x - X).max(axis=1)[0].sum().item()

                    yp_adv, _ = admm_model(adv_x)
                    adv_loss += F.mse_loss(yp_adv.T, y, reduction="sum").item()

                adv_loss /= len(valid_loader.dataset)
            # Accumulate the last results for MSE vs epsilon graph
            final_results_adv[mode].append(adv_loss)
            final_results_clean[mode].append(clean_loss)
            if attack=="CW":
                final_results_l_inf[mode].append(l_inf)

            if save_models and mode != 'admm':
                path = MODEL_PATH_TEMPLATE.format(model='ladmm', attack=attack, mode=mode,N=N,
                                                  epsilon=eps, epochs=num_epochs, MBDL=str(True), K=model.T)

                torch.save(model.state_dict(), path)

            print("mode {0} epsilon {1} ISTA adversarial loss: {2} clean loss {3}".format(mode, eps, adv_loss,
                                                                                          clean_loss))

        #plot_loss_surface_trajectories(robust_model, clean_model, eps, save_figures=save_figures)

    plot_object = {'adv_epsilon_vec': attack_magnitudes, 'final_results_clean': final_results_clean,
                   'final_results_adv': final_results_adv, 'admm_l_inf_cw': final_results_l_inf}

    object_fname = f'LADMM_defense_eps_{str(attack_magnitudes)}_{attack}.pkl'
    save_object(plot_object, object_fname)
    plot_defense_graph(attack_magnitudes, object_fname, 'LADMM')


def load_model_eval_model(path):
    lista_model_temp = LADMM_Model.create_ladmm_model(H, T_ADMM)
    lista_model_temp.load_state_dict(torch.load(path, weights_only=True))
    loaded_model = copy.deepcopy(lista_model_temp)
    loaded_model.eval()
    return loaded_model


def plot_loss_surface_trajectories(robust_model, clean_model, epsilon, save_figures):
    adv_color = 'red'
    clean_color = 'blue'

    # Plotting trajectories upon ISTA_gt loss surface
    x, s = train_loader.dataset[0][0].unsqueeze(0).T, train_loader.dataset[0][1].unsqueeze(0)
    signals = [(x.double(), s.double())]
    kwargs = classic_admm.execute(signals=signals, H_mat=H, plot_graphs=False, get_exec_params_mode=True,
                                  radius_vec=[epsilon])
    robust_model.eval()
    clean_model.eval()

    # TODO - why A, B is using N-number of samples as dimension?
    _, adv_s_hats_traj = robust_model.forward(x.T, acc_s_hat=True)
    _, clean_s_hats_traj = clean_model.forward(x.T, acc_s_hat=True)
    kwargs['clean_trajectory'] = clean_s_hats_traj
    kwargs['adv_trajectory'] = adv_s_hats_traj
    kwargs['steps'] = 20
    kwargs['distance'] = 3
    Z_gt, Z_adv, traj_clean, traj_adv = kwargs['gt_model'].random_plane(**kwargs)
    plt.figure()

    # Plot surface
    cs = plt.contour(Z_gt)
    # Calculate coordianates
    clean_x_cor = [t['i'] for t in traj_clean]
    clean_y_cor = [t['j'] for t in traj_clean]
    adv_x_cor = [t['i'] for t in traj_adv]
    adv_y_cor = [t['j'] for t in traj_adv]

    # Scatter
    plt.scatter(adv_x_cor, adv_y_cor, color=adv_color, marker='x', zorder=5)
    plt.scatter(clean_x_cor, clean_y_cor, color=clean_color, marker='o', zorder=5)
    # Start/Stop text
    plt.text(clean_x_cor[0], clean_y_cor[0], 'Start', fontsize=12, color=clean_color, ha='right', va='bottom')
    plt.text(clean_x_cor[-1], clean_y_cor[-1], 'End', fontsize=12, color=clean_color, ha='right', va='bottom')
    # Start/Stop text
    plt.text(adv_x_cor[0], adv_y_cor[0], 'Start', fontsize=12, color=adv_color, ha='right', va='top')
    plt.text(adv_x_cor[-1], adv_y_cor[-1], 'End', fontsize=12, color=adv_color, ha='right', va='top')
    # plot arrows
    for i in range(len(clean_x_cor) - 1):
        plt.arrow(clean_x_cor[i], clean_y_cor[i], clean_x_cor[i + 1] - clean_x_cor[i],
                  clean_y_cor[i + 1] - clean_y_cor[i], head_width=15, linestyle='--', head_length=15, fc=clean_color,
                  ec=clean_color,
                  length_includes_head=True, zorder=6)
        plt.arrow(adv_x_cor[i], adv_y_cor[i], adv_x_cor[i + 1] - adv_x_cor[i], adv_y_cor[i + 1] - adv_y_cor[i],
                  head_width=15, head_length=15, fc=adv_color, ec=adv_color, linestyle='-.', length_includes_head=True,
                  zorder=6)

    # Styling
    plt.clabel(cs, inline=1, fontsize=10)
    plt.xlabel(r'$u_2$')
    plt.ylabel(r'$u_1$')
    # plt.style.use('plot_style.txt')
    plt.title("ADMM loss surface with trajectories, epsilon={0}".format(epsilon))
    # plt.savefig("ISTA_2D_LOSS_GT.pdf", bbox_inches='tight')
    plt.legend(['LADMM-clean trajectory', 'LADMM-adv trajectory'])
    if save_figures:
        save_fig('ladmm_loss_surface_{0}.pdf'.format(epsilon))
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


if __name__ == '__main__':
    # Train and apply LISTA with T iterations / layers
    ladmm = LADMM_Model.create_ladmm_model(H, T_ADMM)
    #inference(test_loader, save_figures=False)
    for attack in ["NIFGSM", "BIM", "CW"]:

        if attack in ["BIM", "NIFGSM"]:
            attack_magnitudes = [0.005, 0.025, 0.045, 0.065, 0.085]
            # attack_magnitudes = [0.025, 0.045, 0.065, 0.085]
        elif attack == "CW":
            attack_magnitudes = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        else:
            raise Exception("Not implemented attack")
        train(ladmm, train_loader, test_loader, num_epochs=def_num_epochs,
              attack=attack, attack_magnitudes=attack_magnitudes, save_models=True)

    # 1. MLSP
    # 2. RPCA attack
    # 3. Loss plots Robust LISTA vs LISTA vs ISTA
    # 4. Trajectory of LISTA/ LISTA_ADV upon ISTA loss surface
