import torch
from utills import plot_defense_graph
import torch.nn.functional as F
import torchattacks
import torch.nn as nn
import numpy as np
import random
from utills import epoch, epoch_adversarial, save_fig, MODEL_PATH_TEMPLATE, get_attack_func, save_object, load_object
from data_utils import create_data_set
import copy

import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.set_default_dtype(torch.float64)
BATCH_SIZE = 50

N = 1200  # number of samples

# n = 150  # dim(x)
# m = 200  # dim(s)
# k = 4  # k-sparse signal

# Signal generation configuration
m, n, k = 1500, 256, 5

# Measurement matrix
H = torch.randn(n, m)
H /= torch.norm(H, dim=0)

import classic_ista

classic_ista.m = m
classic_ista.N = N
classic_ista.H = H
# Generate datasets
train_loader = create_data_set(H, n=n, m=m, k=k, N=N, batch_size=BATCH_SIZE)
test_loader = create_data_set(H, n=n, m=m, k=k, N=N, batch_size=N//3)


def inference(valid_loader, adv_epsilon_vec, save_figure, epochs, attack):
    """
    validation - for plot recreation
    """

    final_results_adv = {'ista': [], 'clean_model': [], 'robust_model': []}
    final_results_clean = {'ista': [], 'clean_model': [], 'robust_model': []}
    for eps in adv_epsilon_vec:
        # Accumulate history for MSE vs epoch graph
        for mode in ['clean_model', 'robust_model', 'ista']:
            # # Initialization
            if mode in ['clean_model', 'robust_model']:
                # Loading the model
                path = MODEL_PATH_TEMPLATE.format(model='lista', attack=attack, mode=mode, epsilon=eps, epochs=epochs,
                                                  MBDL=str(True), K=LISTA_Model.T_LISTA)
                model = load_model_eval_model(path)

                clean_loss = epoch(valid_loader, model)
                adv_loss = epoch_adversarial(valid_loader, model, attack=attack, eps=eps)

            else:
                ista_model = classic_ista.ISTA.create_ISTA(H=H, max_iter=1000)
                adv_loss, clean_loss = 0., 0.
                for X, y in valid_loader:
                    yp, _ = ista_model(X)
                    clean_loss += F.mse_loss(yp.T, y, reduction="sum").item()
                clean_loss /= len(valid_loader.dataset)

                for X, y in valid_loader:
                    adv_x, _ = classic_ista.BIM(ista_model, X, y, eps=eps)
                    yp_adv, _ = ista_model(adv_x)
                    adv_loss += F.mse_loss(yp_adv.T, y, reduction="sum").item()

                adv_loss /= len(valid_loader.dataset)
            # Accumulate the last results for MSE vs epsilon graph
            final_results_adv[mode].append(adv_loss)
            final_results_clean[mode].append(clean_loss)

            if mode == 'robust_model':
                robust_model = copy.deepcopy(model)
            elif mode == 'clean_model':
                clean_model = copy.deepcopy(model)

            print("mode {0} epsilon {1} ISTA adversarial loss: {2} clean loss {3}".format(mode, eps, adv_loss,
                                                                                          clean_loss))

        plot_loss_surface_trajectories(valid_loader, robust_model, clean_model, eps, save_figure)

    plot_object = {'adv_epsilon_vec': adv_epsilon_vec, 'final_results_clean': final_results_clean,
                   'final_results_adv': final_results_adv}

    object_fname = f'LISTA_defense_eps_{str(adv_epsilon_vec)}_{attack}.pkl'
    save_object(plot_object, object_fname)
    plot_defense_graph(adv_epsilon_vec, object_fname, 'LISTA')


def train(original_model, train_loader, valid_loader, num_epochs, attack, attack_magnitudes, save_models=False,
          save_figures=True):
    """Train a network.
    Returns:
        loss_test {numpy} -- loss function values on test set
    """

    final_results_adv = {'ista': [], 'clean_model': [], 'robust_model': []}
    final_results_clean = {'ista': [], 'clean_model': [], 'robust_model': []}
    final_results_l_inf = {'ista': [], 'clean_model': [], 'robust_model': []}
    # TODO - train LISTA-clean only once

    # adv_epsilon_vec = list(np.linspace(0.006, attack_max_radius, 4))
    for eps in attack_magnitudes:
        # Accumulate history for MSE vs epoch graph
        if attack in ["BIM", "NIFGSM"]:
            attack_kwargs = dict(eps=eps)
        elif attack == "CW":
            attack_kwargs = dict(c=eps)
        else:
            raise Exception("Not implemented attack")

        clean_model_adv, clean_model_clean, robust_model_adv, robust_model_clean = [], [], [], []
        for mode in ['robust_model', 'clean_model', 'ista']:
            # # Initialization
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

                ista_model = classic_ista.ISTA.create_ISTA(H=H, max_iter=1000)
                adv_loss, clean_loss = 0., 0.
                l_inf = 0.
                for X, y in valid_loader:
                    yp, _ = ista_model(X)
                    clean_loss += F.mse_loss(yp.T, y, reduction="sum").item()
                clean_loss /= len(valid_loader.dataset)

                for X, y in valid_loader:
                    adv_x, _ = get_attack_func(attack_name=attack)(ista_model, X, y, **attack_kwargs)
                    if attack=="CW":
                        l_inf += abs(adv_x - X).max(axis=1)[0].sum().item()

                    yp_adv, _ = ista_model(adv_x)
                    adv_loss += F.mse_loss(yp_adv.T, y, reduction="sum").item()
                l_inf /= len(valid_loader.dataset)
                adv_loss /= len(valid_loader.dataset)
            # Accumulate the last results for MSE vs epsilon graph
            final_results_adv[mode].append(adv_loss)
            final_results_clean[mode].append(clean_loss)
            if attack=="CW":
                final_results_l_inf[mode].append(l_inf)

            if save_models and mode != 'ista':
                path = MODEL_PATH_TEMPLATE.format(model='lista', attack=attack, mode=mode, epochs=num_epochs,
                                                  epsilon=eps, MBDL=str(model.A_B_MBDL), K=LISTA_Model.T_LISTA)
                torch.save(model.state_dict(), path)

            print("mode {0} epsilon {1} ISTA adversarial loss: {2} clean loss {3}".format(mode, eps, adv_loss,
                                                                                          clean_loss))

    #     plot_loss_surface_trajectories(valid_loader, robust_model, clean_model, eps, save_figure=save_figures)
    #

    plot_object = {'adv_epsilon_vec': attack_magnitudes, 'final_results_clean': final_results_clean,
                   'final_results_adv': final_results_adv, 'ista_l_inf_cw': final_results_l_inf}

    object_fname = f'LISTA_defense_eps_{str(attack_magnitudes)}_{attack}.pkl'
    save_object(plot_object, object_fname)
    plot_defense_graph(attack_magnitudes, object_fname, 'LISTA')

    # plot_mse_vs_epsilon_graphs(attack_magnitudes, final_results_clean, final_results_adv, save_figure=save_figures)



class LISTA_Model(nn.Module):
    T_LISTA = 5

    def __init__(self, n, m, T, rho=1.0, H=None, A_B_MBDL=True):
        super(LISTA_Model, self).__init__()
        self.n, self.m = n, m
        self.H = H
        self.T = T  # ISTA Iterations
        self.rho = rho  # Lagrangian Multiplier
        self.A = nn.Linear(n, m, bias=False)  # Weight Matrix
        self.B = nn.Linear(m, m, bias=False)  # Weight Matrix
        self.A_B_MBDL = A_B_MBDL

        if not self.A_B_MBDL:
            for p in self.A.parameters():
                p.requires_grad = False

            for p in self.B.parameters():
                p.requires_grad = False

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


def load_model_eval_model(path):
    lista_model_temp = LISTA_Model.create_lista_model()
    lista_model_temp.load_state_dict(torch.load(path, weights_only=True))
    loaded_model = copy.deepcopy(lista_model_temp)
    loaded_model.eval()
    return loaded_model


def plot_bound_graph(adv_epsilon_vec):
    # Calculate ISTA bound
    # avg_number_ista_iterations = 300
    # ista_obj = classic_ista.ISTA.create_ISTA()
    # B_ista = ista_obj.mu*torch.svd(ista_obj.H.T).S.max()
    # tmp = ista_obj.mu*ista_obj.H.T@ista_obj.H
    # M_ista = torch.svd(torch.eye(tmp.shape[0], tmp.shape[1]) - tmp).S.max()
    #
    # C_ista = 0
    # for t in range(0, avg_number_ista_iterations):
    #     prod_sum_ista_t = 1
    #     for j in range(t+1, avg_number_ista_iterations):
    #         prod_sum_ista_t*=M_ista
    #
    #     C_ista += prod_sum_ista_t*B_ista

    cl_sing_max, adv_sing_max, classic_ista_sing_max = [], [], []
    for e in adv_epsilon_vec:
        cl_max_singular_value, adv_max_singular_value = calculate_bound(e)
        cl_sing_max.append(cl_max_singular_value.item())
        adv_sing_max.append(adv_max_singular_value.item())

    norm_factor = max(cl_sing_max)
    cl_sing_max = np.array(cl_sing_max) / norm_factor
    adv_sing_max = np.array(adv_sing_max) / norm_factor

    save_object({'LISTA': cl_sing_max, 'Robust-LISTA': adv_sing_max}, 'bound_graph_n=1200.pkl')


def calculate_bound(eps):
    # Calculate LISTA bounds
    path_clean = MODEL_PATH_TEMPLATE.format(model='lista', mode='clean_model', epsilon=eps, epochs=epochs, attack='BIM',
                                            MBDL=str(True), K=LISTA_Model.T_LISTA)

    path_adv = MODEL_PATH_TEMPLATE.format(model='lista', mode='robust_model', epsilon=eps, epochs=epochs, attack='BIM',
                                          MBDL=str(True), K=LISTA_Model.T_LISTA)

    lista_clean = load_model_eval_model(path_clean)
    lista_robust = load_model_eval_model(path_adv)

    M_i_cl = [torch.svd(torch.eye(lista_clean.B.weight.shape[0]) - mu_i * lista_clean.B.weight).S.max() for mu_i in
              list(lista_clean.mu)]
    B_i_cl = [mu_i * torch.svd(lista_clean.A.weight).S.max() for mu_i in list(lista_clean.mu)]

    M_i_adv = [torch.svd(torch.eye(lista_robust.B.weight.shape[0]) - mu_i * lista_robust.B.weight).S.max() for mu_i in
               list(lista_robust.mu)]
    B_i_adv = [mu_i * torch.svd(lista_robust.A.weight).S.max() for mu_i in list(lista_robust.mu)]

    C_cl = 0
    C_adv = 0
    for t in range(0, LISTA_Model.T_LISTA):
        prod_sum_cl_t = 1
        prod_sum_adv_t = 1
        for j in range(t + 1, LISTA_Model.T_LISTA):
            prod_sum_cl_t *= M_i_cl[j]
            prod_sum_adv_t *= M_i_adv[j]

        C_cl += prod_sum_cl_t * B_i_cl[t]
        C_adv += prod_sum_adv_t * B_i_adv[t]
    return C_cl, C_adv


def plot_loss_surface_trajectories(validation_loader, robust_model, clean_model, epsilon, save_figure=False):
    adv_color = 'red'
    clean_color = 'blue'

    # Plotting trajectories upon ISTA_gt loss surface
    x, s = validation_loader.dataset[0][0].unsqueeze(0).T, validation_loader.dataset[0][1].unsqueeze(0)
    kwargs = classic_ista.execute(signals=[(x.double(), s.double())], H_mat=H,
                                  plot_graphs=False, get_exec_params_mode=True, radius_vec=[epsilon])
    robust_model.eval()
    clean_model.eval()

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
    # transform coordiantes
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
    plt.grid(True)
    plt.grid(True)
    plt.legend(['LISTA', 'Robust-LISTA'])
    if save_figure:
        save_fig('LISTA_trajectories_{0}.pdf'.format(epsilon))
    plt.show()

def plot_trajectory_paper_graph():
    # TRAJECTORY
    path_cl = MODEL_PATH_TEMPLATE.format(model='lista', attack="BIM", mode="clean_model",
                                      epsilon=0.005, epochs=40,
                                      MBDL=str(True), K=5)

    path_adv = MODEL_PATH_TEMPLATE.format(model='lista', attack="BIM", mode="robust_model",
                                      epsilon=0.005, epochs=40,
                                      MBDL=str(True), K=5)
    clean = load_model_eval_model(path_cl)
    adv = load_model_eval_model(path_adv)
    plot_loss_surface_trajectories(test_loader, adv,clean,epsilon=0.025,save_figure=True)



if __name__ == '__main__':

    # Train and apply LISTA with T iterations / layers

    lista = LISTA_Model.create_lista_model()
    epochs = 40

    # Plot trajectory graph
    plot_trajectory_paper_graph()

    # adv_epsilon_vec = [0.005, 0.025, 0.045, 0.065, 0.085]

    # Attacks = BIM/CW/FGSM-NITRO

    # for attack in ["NIFGSM", "BIM"]:
    #
    #     if attack in ["BIM", "NIFGSM"]:
    #         attack_magnitudes = [0.005, 0.025, 0.045, 0.065, 0.085]
    #         # attack_magnitudes = [0.025, 0.045, 0.065, 0.085]
    #     elif attack == "CW":
    #         attack_magnitudes = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    #     else:
    #         raise Exception("Not implemented attack")

        #inference(valid_loader=test_loader, adv_epsilon_vec=attack_magnitudes, attack=attack, save_figure=True, epochs=epochs)
        # train(lista, train_loader, test_loader, attack=attack,
        #       attack_magnitudes=attack_magnitudes, num_epochs=epochs,
        #       save_models=True, save_figures=True)


    # JL lemma - Johnson Lindenshtauch Lemma- norm0 vs norm 1
