import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
from utills import epoch, epoch_adversarial, save_fig, MODEL_PATH_TEMPLATE
from data_utils import create_data_set

import copy

import matplotlib.pyplot as plt


SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.set_default_dtype(torch.float64)
BATCH_SIZE = 50
adv_epsilon_vec = [0.005, 0.025, 0.045, 0.065, 0.085]

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
test_loader = create_data_set(H, n=n, m=m, k=k, N=N, batch_size=N)

attack_radius = 0.9
epochs = 40

def inference(valid_loader, adv_epsilon_vec, save_figure,epochs):
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
                path = MODEL_PATH_TEMPLATE.format(model='lista', mode=mode, epsilon=eps, epochs=epochs,
                                                  MBDL=str(True), K=LISTA_Model.T_LISTA)
                model = load_model_eval_model(path)

                clean_loss = epoch(valid_loader, model)
                adv_loss = epoch_adversarial(valid_loader, model, classic_ista.BIM, eps=eps)

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

        plot_loss_surface_trajectories(valid_loader,robust_model,clean_model,eps,save_figure)

    plot_mse_vs_epsilon_graphs(adv_epsilon_vec,final_results_clean, final_results_adv, save_figure)

def train(original_model, train_loader, valid_loader, num_epochs, attack_max_radius, save_models=False, save_figures=True):
    """Train a network.
    Returns:
        loss_test {numpy} -- loss function values on test set
    """

    final_results_adv = {'ista': [], 'clean_model': [], 'robust_model': []}
    final_results_clean = {'ista': [], 'clean_model': [], 'robust_model': []}
    #adv_epsilon_vec = list(np.linspace(0.006, attack_max_radius, 4))
    for eps in adv_epsilon_vec:
        # Accumulate history for MSE vs epoch graph
        clean_model_adv, clean_model_clean, robust_model_adv, robust_model_clean = [], [], [], []
        for mode in ['ista', 'clean_model', 'robust_model']:
            # # Initialization
            if mode in ['clean_model', 'robust_model']:
                model = copy.deepcopy(original_model)
                optimizer = torch.optim.SGD(model.parameters(), lr=5e-05, momentum=0.9, weight_decay=0)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

                for t in range(num_epochs):
                    model.train()
                    if mode == 'robust_model':
                        train_loss = epoch_adversarial(train_loader, model, classic_ista.BIM, opt=optimizer,
                                                       scheduler=scheduler, eps=eps)
                    else:
                        train_loss = epoch(train_loader, model, opt=optimizer, scheduler=scheduler)

                    # Testing phase - Test upon clean & adversarial test examples
                    model.eval()
                    clean_loss = epoch(valid_loader, model)
                    adv_loss = epoch_adversarial(valid_loader, model, classic_ista.BIM, eps=eps)
                    if mode == 'robust_model':
                        robust_model_adv.append(adv_loss)
                        robust_model_clean.append(clean_loss)
                    else:
                        clean_model_adv.append(adv_loss)
                        clean_model_clean.append(clean_loss)

                    print(*("{:.6f}".format(i) for i in (train_loss, clean_loss, adv_loss)), sep="\t")
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

            if save_models and mode != 'ista':
                path = MODEL_PATH_TEMPLATE.format(model='lista', mode=mode, epochs=num_epochs,
                                                  epsilon=eps,MBDL=str(model.A_B_MBDL),K=LISTA_Model.T_LISTA)
                torch.save(model.state_dict(), path)

            print("mode {0} epsilon {1} ISTA adversarial loss: {2} clean loss {3}".format(mode, eps, adv_loss,
                                                                                          clean_loss))

        plot_loss_surface_trajectories(valid_loader,robust_model, clean_model, eps, save_figure=save_figures)

    plot_mse_vs_epsilon_graphs(adv_epsilon_vec,final_results_clean, final_results_adv,save_figure=save_figures)



class LISTA_Model(nn.Module):
    T_LISTA = 5

    def __init__(self, n, m, T=6, rho=1.0, H=None, A_B_MBDL=True):
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
    lista_model_temp.load_state_dict(torch.load(path,weights_only=True))
    loaded_model = copy.deepcopy(lista_model_temp)
    loaded_model.eval()
    return loaded_model

def plot_bound_graph():

    cl_sing_max, adv_sing_max = [], []
    for e in adv_epsilon_vec:
        cl_max_singular_value, adv_max_singular_value = calculate_bound(e)
        cl_sing_max.append(cl_max_singular_value.item())
        adv_sing_max.append(adv_max_singular_value.item())

    plt.figure()
    plt.title(f'LISTA K={LISTA_Model.T_LISTA} bound')
    plt.plot(adv_epsilon_vec, cl_sing_max)
    plt.plot(adv_epsilon_vec, adv_sing_max)
    plt.legend(['clean-model', 'robust-model'])
    plt.ylabel('max singular value')
    plt.xlabel('epsilon')
    plt.show()

def calculate_bound(eps):
    path_clean = MODEL_PATH_TEMPLATE.format(model='lista', mode='clean_model', epsilon=eps, epochs=epochs,
                                            MBDL=str(True), K=LISTA_Model.T_LISTA)
    path_adv = MODEL_PATH_TEMPLATE.format(model='lista', mode='robust_model', epsilon=eps, epochs=epochs,
                                          MBDL=str(True), K=LISTA_Model.T_LISTA)

    lista_clean = load_model_eval_model(path_clean)
    lista_robust = load_model_eval_model(path_adv)

    A_i_cl = [torch.eye(lista_clean.B.weight.shape[0]) - mu_i * lista_clean.B.weight for mu_i in list(lista_clean.mu)]
    B_i_cl = [mu_i * lista_clean.A.weight for mu_i in list(lista_clean.mu)]

    A_i_adv = [torch.eye(lista_robust.B.weight.shape[0]) - mu_i * lista_robust.B.weight for mu_i in
              list(lista_robust.mu)]
    B_i_adv = [mu_i * lista_robust.A.weight for mu_i in list(lista_robust.mu)]

    delta_s_prev_cl = B_i_cl[0]
    for i in range(1, LISTA_Model.T_LISTA):
        delta_s_curr_cl = A_i_cl[i] @ delta_s_prev_cl + B_i_cl[i]
        delta_s_prev_cl = delta_s_curr_cl.detach()

    # Compute max singular value
    singular_values = torch.svd(delta_s_curr_cl, ).S
    cl_max_singular_value = singular_values.max()


    delta_s_prev_adv = B_i_adv[0]
    for i in range(1, LISTA_Model.T_LISTA):
        delta_s_curr_adv = A_i_adv[i] @ delta_s_prev_adv + B_i_adv[i]
        delta_s_prev_adv = delta_s_curr_adv.detach()

    # Compute max singular value
    singular_values = torch.svd(delta_s_curr_adv, ).S
    adv_max_singular_value = singular_values.max()
    return cl_max_singular_value, adv_max_singular_value



def plot_mse_vs_epsilon_graphs(adv_epsilon_vec, final_results_clean, final_results_adv, save_figure=False):
    plt.figure()
    plt.title('BIM max Epsilon {0}'.format(adv_epsilon_vec[-1]))
    plt.plot(adv_epsilon_vec, final_results_adv['robust_model'], label='robust-model-adv-data', color='b', linewidth=1)
    plt.plot(adv_epsilon_vec, final_results_adv['clean_model'], label='clean_model-adv-data', color='r', linewidth=1)
    plt.plot(adv_epsilon_vec, final_results_adv['ista'], label='ista-adv-data', color='g', linewidth=1)
    plt.xlabel('epsilon')
    plt.ylabel('MSE')
    plt.legend()
    if save_figure:
        save_fig('LISTA_MSE_adv_data.pdf')
    plt.show()

    plt.figure()
    plt.title('BIM max Epsilon {0}'.format(adv_epsilon_vec[-1]))
    plt.plot(adv_epsilon_vec, final_results_clean['robust_model'], label='robust-model-clean-data', color='b',
             linewidth=1)
    plt.plot(adv_epsilon_vec, final_results_clean['clean_model'], label='clean_model-clean-data', color='r',
             linewidth=1)
    plt.plot(adv_epsilon_vec, final_results_clean['ista'], label='ista-clean-data', color='g', linewidth=1)
    plt.xlabel('epsilon')

    plt.ylabel('MSE')
    plt.legend()
    if save_figure:
        save_fig('LISTA_MSE_clean_data.pdf')
    plt.show()

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
    plt.colorbar(cs)
    plt.xlabel(r'$u_2$')
    plt.ylabel(r'$u_1$')
    # plt.style.use('plot_style.txt')
    plt.title("ISTA loss surface with trajectories, epsilon={0}".format(epsilon))
    # plt.savefig("ISTA_2D_LOSS_GT.pdf", bbox_inches='tight')
    plt.legend(['LISTA-clean trajectory', 'LISTA-adv trajectory'])
    if save_figure:
        save_fig('lista_loss_surface_{0}.pdf'.format(epsilon))
    plt.show()


if __name__ == '__main__':

    T_LISTA = LISTA_Model.T_LISTA = 5

    # Train and apply LISTA with T iterations / layers
    lista = LISTA_Model.create_lista_model()
    # inference(valid_loader=test_loader, adv_epsilon_vec=adv_epsilon_vec, save_figure=True, epochs=epochs)
    # plot_bound_graph()
    train(lista, train_loader, test_loader, num_epochs=epochs,
          attack_max_radius=attack_radius, save_models=True, save_figures=False)


    # 1. MLSP
    # 2. RPCA attack
    # 3. Loss plots Robust LISTA vs LISTA vs ISTA
    # 4. Trajectory of LISTA/ LISTA_ADV upon ISTA loss surface

    # JL lemma - Johnson Lindenshtauch Lemma- norm0 vs norm 1
