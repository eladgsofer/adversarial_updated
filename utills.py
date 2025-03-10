__author__ = 'Elad Sofer <elad.g.sofer@gmail.com>'

import os

import torch
import torch.nn as nn
import seaborn as sns


import torchattacks
import numpy as np
from scipy.interpolate import interp1d
import copy
import pickle

import torch.nn.functional as F
from scipy.linalg import orth
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

set_matplotlib_formats('svg', 'pdf')
matplotlib.rcParams['lines.linewidth'] = 2.0

np.random.seed(0)
# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

"""
This module comprises various utility functions for the project such as:
 1. Plotting utilities
 2. BIM attack implementation
 3. dataset generation functions
 4. configuration parameters.
"""

FIGURES_PATH = r'data/graphs/'
MATRICES_PATH = r'data/matrices/'
MODELS_PATH = r'data/models_history/'
PKL_PATH = r'data/graph_pkl_files'

MODEL_PATH_TEMPLATE = os.path.join(MODELS_PATH,
                                   '{model}_{attack}_{mode}_{epsilon}_epochs_{epochs}_MBDL_{MBDL}_K={K}.npy')
# Attack configuration
r_step = 40
sig_amount = 1
eps_min, eps_max = 0.01 * 0.5, 0.05

# Loss-surface visualization resolution
loss3d_res_steps = 800

# Signal generation configuration
m, n, k = 1500, 256, 5
Psi = np.eye(m)
Phi = np.random.randn(n, m)
Phi = np.transpose(orth(np.transpose(Phi)))
H = Phi
H = torch.from_numpy(H).float()


def generate_signal():
    """
    Generate a sparse signal 's' and its observation 'x' using the model x = Hs + w, where w is a Gaussian noise.
    """
    s = np.zeros((1, m))
    index_k = np.random.choice(m, k, replace=False)
    s[:, index_k] = 0.5 * np.random.randn(k, 1).reshape([1, k])
    s = torch.from_numpy(s).float()

    # x = Hs+w s.t w~N(0,1)
    x = np.dot(H, s.T) + 0.01 * np.random.randn(n, 1)
    x = torch.from_numpy(x).float()
    return x.detach(), s.detach()


def CarliniWagner(model, x, s_gt, c):
    cw_obj = torchattacks.CW(model, c=c)
    return cw_obj(x, s_gt)


def NIFGSM(model, x, s_gt, eps=0.1, alpha=0.01, steps=5, decay=1, randomize=True, **kwargs):
    """
    The BIM (Basic Iterative Method) adversarial attack is a technique used to generate adversarial examples usually
     for machine learning models. This function aims to attack ADMM/ISTA optimizers.
    :param model: ADMM/ISTA object, the target machine learning model to be attacked.
    :param x: torch vector x - x=Hs+w s.t w~N(0,0.001)
    :param s_gt: torch vector which represents s^*
    :param eps:   A small perturbation magnitude that controls the strength of the attack
    :param alpha: A step size parameter for adjusting the perturbation at each iteration
    :param steps: The number of iterations to perform the attack.
    :return: adversarial x signal and the pertubation which was applied.
    """
    # prev_conf = model.require_grad
    # model.require_grad = False
    # batch = False if 1 in x.shape else True
    x = x.clone().to(device)
    s_gt = s_gt.clone().to(device)
    momentum = torch.zeros_like(x).detach()

    loss = nn.MSELoss()

    original_x = x.data
    adv_x = x.clone().detach()

    if randomize:
        delta = torch.rand_like(adv_x, requires_grad=False)
        delta.data = delta.data * 2 * eps - eps
        delta.data = delta.clamp(-eps, eps)
    else:
        delta = torch.zeros_like(adv_x, requires_grad=False)

    adv_x = adv_x + delta
    for step in range(steps):
        # print("BIM Step {0}".format(step))

        adv_x.requires_grad = True
        nes_x = adv_x + decay * alpha * momentum
        s_hat, _ = model(nes_x, **kwargs)
        model.zero_grad()

        # Calculate loss
        if s_gt.shape != s_hat.shape:
            cost = loss(s_gt, s_hat.T)
        else:
            cost = loss(s_gt, s_hat)

        # cost.backward(retain_graph=True)
        grad = torch.autograd.grad(cost, adv_x)[0]
        grad = decay * momentum + grad  # /torch.mean(torch.abs(grad), dim=(1), keepdim=True)
        momentum = grad

        # Grad is calculated
        delta = alpha * grad.sign()

        # Stop following gradient changes
        adv_x = adv_x.clone().detach()

        adv_x = adv_x + delta

        # Clip the change between the adversarial images and the original images to an epsilon range
        eta = torch.clamp(adv_x - original_x, min=-eps, max=eps)

        adv_x = original_x + eta

    return adv_x, delta  # grad is the gradient (perturbation)


def BIM(model, x, s_gt, eps=0.1, alpha=0.01, steps=5, randomize=True, **kwargs):
    """
    The BIM (Basic Iterative Method) adversarial attack is a technique used to generate adversarial examples usually
     for machine learning models. This function aims to attack ADMM/ISTA optimizers.
    :param model: ADMM/ISTA object, the target machine learning model to be attacked.
    :param x: torch vector x - x=Hs+w s.t w~N(0,0.001)
    :param s_gt: torch vector which represents s^*
    :param eps:   A small perturbation magnitude that controls the strength of the attack
    :param alpha: A step size parameter for adjusting the perturbation at each iteration
    :param steps: The number of iterations to perform the attack.
    :return: adversarial x signal and the pertubation which was applied.
    """
    # prev_conf = model.require_grad
    # model.require_grad = False
    # batch = False if 1 in x.shape else True
    x = x.clone().to(device)
    s_gt = s_gt.clone().to(device)

    loss = nn.MSELoss()

    original_x = x.data
    adv_x = x.clone().detach()

    if randomize:
        delta = torch.rand_like(adv_x, requires_grad=False)
        delta.data = delta.data * 2 * eps - eps
        delta.data = delta.clamp(-eps, eps)
    else:
        delta = torch.zeros_like(adv_x, requires_grad=False)

    adv_x = adv_x + delta
    for step in range(steps):
        # print("BIM Step {0}".format(step))

        adv_x.requires_grad = True
        s_hat, _ = model(adv_x, **kwargs)
        model.zero_grad()

        # Calculate loss
        if s_gt.shape != s_hat.shape:
            cost = loss(s_gt, s_hat.T)
        else:
            cost = loss(s_gt, s_hat)

        # cost.backward(retain_graph=True)
        grad = torch.autograd.grad(cost, adv_x)[0]

        # Grad is calculated
        delta = alpha * grad.sign()

        # Stop following gradient changes
        adv_x = adv_x.clone().detach()

        adv_x = adv_x + delta

        # Clip the change between the adversarial images and the original images to an epsilon range
        eta = torch.clamp(adv_x - original_x, min=-eps, max=eps)

        adv_x = original_x + eta

    # model.require_grad = prev_conf
    return adv_x, delta  # grad is the gradient (perturbation)


def plot_1d_surface(gt_line, adv_line, fname):
    """
    The plot_loss_surfaces_u1 function is used to plot the loss surfaces along the u1 axis in a 1D dimension.
    To understand more about the visualization process, refer to the linear_interpolation function under the visualize_model module.
    :param gt_line: a vector which includes loss surface samples along u1 axis of L_{op}
    :param adv_line: a vector which includes loss surface samples along u1 axis of L_{adv}
    fname: The file name to save the figure to.
    :return:
    """
    plt.figure()
    plt.plot(np.arange(len(gt_line)), gt_line)
    plt.plot(np.arange(len(adv_line)), adv_line)
    plt.xlabel(r'$u_1$')
    plt.legend([r'Loss $\mathcal{L}_{op}$', r'Loss $\mathcal{L}_{adv}$'])
    save_fig(fname)
    plt.show()


def plot_2d_surface(z_gt, z_adv, fname):
    """
    This plot_2d_surface function is used to plot contour plots of the loss surfaces instead of 3D plots as
    function plot_3d_surface executes
     To understand more about the visualization process, refer to the random_plane function under the visualize_model module.
    :param z_adv: Grid of loss values for L_adv.
    :param z_gt: Grid of loss values for L_op.
    :param steps: The dimension of the grids Z_adv and Z_gt.
    :param fname: The file name to save the figure to.
    """
    plt.figure()
    cs = plt.contour(z_gt)
    plt.clabel(cs, inline=1, fontsize=10)
    plt.colorbar(cs)
    plt.xlabel(r'$u_2$')
    plt.ylabel(r'$u_1$')
    # plt.style.use('plot_style.txt')
    plt.title("Loss surface of L_truth(s) = 0.5*||x-Hs|| + rho*||s| s.t (rho=0.01), epsilon=0.1")
    plt.savefig("ISTA_2D_LOSS_GT.pdf", bbox_inches='tight')

    plt.figure()
    cs = plt.contour(z_adv)
    plt.clabel(cs, inline=1, fontsize=10)
    plt.colorbar(cs)
    plt.xlabel(r'$u_2$')
    plt.ylabel(r'$u_1$')
    # plt.style.use('plot_style.txt')
    save_fig(fname)
    plt.show()


def plot_3d_surface(z_adv, z_gt, steps, fname):
    """
    This plot_3d_surface function is used to plot Figure 4,
    which displays 3D loss surfaces L_adv and L_op.
     To understand more about the visualization process, refer to the random_plane function under the visualize_model module.
    :param z_adv: Grid of loss values for L_adv.
    :param z_gt: Grid of loss values for L_op.
    :param steps: The dimension of the grids Z_adv and Z_gt.
    :param fname: The file name to save the figure to.
    """
    x, y = np.arange(0, steps), np.arange(0, steps)
    x_vec, y_vec = np.meshgrid(x, y)

    # Plotting 3D
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    plt.style.use('default')
    # plt.axes(projection='3d')
    axs[0].view_init(30, 35)
    axs[0].contour3D(x_vec / 800, y_vec / 800, z_adv, 50, cmap='binary')
    axs[0].set_xlabel(r'$u_2$')
    axs[0].set_ylabel(r'$u_1$')
    axs[0].set_zlabel(r'Loss $\mathcal{L}_{adv}$')
    # plt.title("Loss_adv = 0.5*||x_Adv-Hs_adv|| + rho*||s_adv| s.t (rho=0.01), epsilon=0.1")

    new_pos = axs[1].get_position()
    new_pos.x0 += 0.08 * new_pos.x0
    new_pos.x1 += 0.08 * new_pos.x1
    axs[1].set_position(pos=new_pos)
    axs[1].contour3D(x_vec / 800, y_vec / 800, z_gt, 50, cmap='binary')
    axs[1].set_xlabel(r'$u_2$')
    axs[1].set_ylabel(r'$u_1$')
    axs[1].set_zlabel(r'Loss $\mathcal{L}_{op}$')
    # plt.title("Loss_gt = 0.5*||x-Hs|| + rho*||s| s.t (rho=0.01), epsilon=0.1")
    axs[1].view_init(30, 35)
    # plt.style.use('plot_style.txt')
    save_fig(fname)
    plt.show()


def plot_conv_rec_graph(signal_a, signal_b, s_gt, errors_a, errors_b,
                        fname="convergence_ADMM.pdf"):
    """
    This function plot_figure3 is used to plot Figure 3. It creates a figure with two subplots.
     The first subplot displays three sparsed signals: signal_a, signal_b, and s_gt (ground truth signal).
      The second subplot shows the convergence errors for signal_a and signal_b over the iterations.

    :param signal_a: Sparse signal vector for Signal A.
    :param signal_b: Sparse signal vector for Signal B.
    :param s_gt: Sparse signal vector for the ground truth signal.
    :param errors_a: Sparse signal vector for the ground truth signal.
    :param errors_b: Error vector holding the errors along the convergence process for Signal B.
    :param fname: the file name to save the figure to
    """
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    # plt.grid(color='gray')
    plt.plot(errors_a, label=r'${s}_{\rm adv}^{\star}$ convergence', linewidth=2)
    plt.plot(errors_b, '--', label=r'${s}^{\star}$ convergence', linewidth=2)
    plt.grid()
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('$\mathcal{L}$', fontsize=13)
    plt.legend()
    # plt.style.use('plot_style.txt')
    plt.subplot(2, 1, 2)
    plt.grid()
    plt.plot(signal_a, '--*', label=r'${s}_{\rm adv}^{\star}$', color='k', linewidth=1)
    plt.plot(signal_b, '-s', label=r'${s}^{\star}$', color='r', linewidth=1)
    plt.plot(s_gt[0], '.-', label="$s$", color='g', linewidth=1)
    plt.xlabel('Index', fontsize=13)
    plt.ylabel('Value', fontsize=13)
    plt.legend()
    save_fig(fname)
    plt.show()


def plot_norm_graph(radius_vec, min_dist, fname):
    """
    Plots the norm graph, showing the relationship between the radius (epsilon) and the minimum distance between
    the optimal signal and the adversarial signal.

    :param radius_vec: A vector of radius values (epsilon).
    :param min_dist: A vector of corresponding minimum distances between the optimal signal and the adversarial signal.
    :param fname: The file name to save the figure to.
    """
    plt.figure()
    plt.plot(radius_vec, min_dist)
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'${\|\| {s}^{\star} - {s}_{\rm adv}^{\star} \|\|}_2$')
    save_fig(fname)
    plt.show()


def plot_observations(adv_x, x, fname):
    """
    Plots the observations of x_{adv} and x signals in two subplots.
     The first subplot shows the values of adv_x as a black line, while the second subplot shows the values of x as a red line.
      Both subplots have grids, x-axis labels ("Index"), and y-axis labels ("Value"). The legend indicates the line corresponding to each subplot.
      The second subplot is slightly adjusted in position to make room for the legend.
       Finally, the figure is saved as fname and displayed.
    :param adv_x: Tensor containing the values of adv_x.
    :param: Tensor containing the values of x.
    :param fname: File name to save the figure.

    :return:
    """
    plt.style.use('default')
    plt.figure()

    plt.subplot(2, 1, 1)
    plt.grid()
    plt.xlabel('Index', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.plot(adv_x.numpy(), color='k')
    plt.legend([r"$x + \delta $"])

    axs1 = plt.subplot(2, 1, 2)
    plt.plot(x.numpy(), color='r')
    plt.grid()
    plt.legend([r"$x$"])
    new_pos = axs1.get_position()
    new_pos.y0 -= 0.08 * new_pos.y0
    new_pos.y1 -= 0.08 * new_pos.y1
    axs1.set_position(pos=new_pos)
    plt.xlabel('Index', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    save_fig(fname)
    plt.show()


def save_fig(fname, transparent=True):
    """
    The function saves the current figure to the specified file by using plt.savefig with the file path obtained by joining FIGURES_PATH and fname
    :param fname:  File name or path to save the figure.
    """

    plt.savefig(os.path.join(FIGURES_PATH, fname), bbox_inches='tight', format='pdf', transparent=transparent)




def save_object(object, fname):
    with open(os.path.join(PKL_PATH, fname), 'wb') as fd:
        pickle.dump(object, fd)

def load_object(fname):
    with open(os.path.join(PKL_PATH, fname), 'rb') as fd:
        return pickle.load(fd)


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


def calculate_bound_single_model(model, delta):
    lista_model = copy.deepcopy(model)
    lista_model.eval()
    M_i = [torch.eye(lista_model.B.weight.shape[0]) - mu_i * lista_model.B.weight for mu_i in list(lista_model.mu)]
    b_i = [mu_i * lista_model.A.weight for mu_i in list(lista_model.mu)]

    delta_s_prev_cl = b_i[0] @ delta
    for i in range(1, len(list(lista_model.mu))):
        delta_s_curr_cl = M_i[i] @ delta_s_prev_cl + b_i[i] @ delta
        delta_s_prev_cl = delta_s_curr_cl.detach()

    return delta_s_curr_cl


def get_attack_func(attack_name):
    if type(attack_name)!=str:
        return attack_name
    if attack_name == "CW":
        return CarliniWagner
    elif attack_name == "BIM":
        return BIM
    elif attack_name == "NIFGSM":
        return NIFGSM
    else:
        raise Exception("not valid attack")


def epoch_adversarial(loader, model, attack, opt=None, scheduler=None, **attack_kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0., 0.
    l_inf_total = 0.
    attack_method = get_attack_func(attack_name=attack)

    for X, y in loader:
        adv_x, _ = attack_method(model, X, y, **attack_kwargs)
        if attack=="CW":
            l_inf_total+= abs(adv_x - X).max(axis=1)[0].sum().item()
        yp, e_loss = model(adv_x)
        # import matplotlib.pyplot as plt
        # plt.plot(X[0].detach())
        # plt.plot(adv_x[0].detach())
        # plt.show()
        loss = F.mse_loss(yp, y, reduction="sum")
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_loss += loss.data.item()
    if scheduler:
        scheduler.step()
    return total_loss / len(loader.dataset), l_inf_total/len(loader.dataset)

def plot_paper_graph(x, result_object, graph_fname,
                     xlabel=r'$\epsilon$', ylabel=r'${\| {s}^{\star} - {s}_{\rm adv}^{\star} \|}_2$',
                     interpol_steps=150, load=True, interpolation=False, x_logscale=False, y_logscale=False):
    sns.set_style("whitegrid")  # Use a white background with gridlines
    styling = ['-', ':', '--', '-.', (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]
    X_ = np.linspace(x[0], x[-1], interpol_steps)
    if load:
        results = load_object(result_object)
    else:
        results = result_object
    titles = []
    plt.figure()
    for idx, (title, y) in enumerate(results.items()):
        titles.append(title)
        if interpolation:
            cubic_interpolation_model = interp1d(x, y, kind="cubic")
            Y_ = cubic_interpolation_model(X_)
            plt.plot(X_, Y_, linestyle=styling[idx])
        else:
            plt.plot(x, y, linestyle=styling[idx])


    plt.legend(titles)
    plt.xlabel(xlabel)
    if x_logscale:
        plt.xscale('log')
    if y_logscale:
        plt.yscale('log')
    plt.ylabel(ylabel)
    plt.grid(True)
    save_fig(graph_fname)
    plt.show()


def plot_defense_graph(x, res_dict_path, algorithm,attack, **kwargs):

    res_object = load_object(res_dict_path)
    if type(x)==str:
        x = res_object[x]

    clean_results = {algorithm: res_object['final_results_clean']['clean_model'],
                     f'Robust-{algorithm}': res_object['final_results_clean']['robust_model'],
                     str(algorithm[1:]): res_object['final_results_clean'][algorithm[1:].lower()]}

    adv_results = {algorithm: res_object['final_results_adv']['clean_model'],
                   f'Robust-{algorithm}': res_object['final_results_adv']['robust_model'],
                   str(algorithm[1:]): res_object['final_results_adv'][algorithm[1:].lower()]}

    plot_paper_graph(x, clean_results, f"defense_clean_data_{algorithm}_{attack}.pdf",ylabel='$\|\hat{s}-s^*\|_2$', **kwargs)
    plot_paper_graph(x, adv_results, f"defense_adv_data_{algorithm}_{attack}.pdf", **kwargs)

if __name__ == '__main__':
    attack_radius_vec = [0.005, 0.025, 0.045, 0.065, 0.085]

    # # # Carlini-Wagner Defense - LISTA
    lista_cw_path = r"/Users/eladsofer/venv/icml/adversarial_updated/data/graph_pkl_files/for paper/defenses/LISTA_defense_eps_[1e-05, 0.0001, 0.001, 0.01, 0.1, 1]_CW.pkl"
    plot_defense_graph(x='adv_epsilon_vec', res_dict_path=lista_cw_path, algorithm='LISTA', xlabel='$c$', attack='CW',
                       load=False, interpolation=False, x_logscale=True)

    # # BIM - Defense - LISTA
    lista_bim_path = r"/Users/eladsofer/venv/icml/adversarial_updated/data/graph_pkl_files/for paper/defenses/LISTA_defense_eps_[0.005, 0.025, 0.045, 0.065, 0.085]_BIM.pkl"
    plot_defense_graph(attack_radius_vec, lista_bim_path, attack="BIM", algorithm='LISTA',
                       load=False, interpolation=True)
    # # NIFGSM - Defense - LISTA
    lista_nifgsm_path = r"/Users/eladsofer/venv/icml/adversarial_updated/data/graph_pkl_files/for paper/defenses/LISTA_defense_eps_[0.005, 0.025, 0.045, 0.065, 0.085]_NIFGSM.pkl"
    plot_defense_graph(attack_radius_vec, lista_nifgsm_path, attack="NIFGSM", algorithm='LISTA',
                       load=False, interpolation=True)

    # LISTA Bound-Graph
    plot_paper_graph(attack_radius_vec,
                     "/Users/eladsofer/venv/icml/adversarial_updated/data/graph_pkl_files/for paper/bound_graph_n=1200.pkl",
                     'LISTA_bound_graph.pdf', ylabel=r'$C(\theta)}$',interpolation=True)

    # BIM Defense - LADMM
    ladmm_bim_path = r"/Users/eladsofer/venv/icml/adversarial_updated/data/graph_pkl_files/LADMM_defense_eps_[0.005, 0.025, 0.045, 0.065, 0.085]_BIM.pkl"
    plot_defense_graph(attack_radius_vec, ladmm_bim_path, attack="BIM",
                       algorithm='LADMM', load=False, interpolation=True)

    ladmm_nifgsm_path = r"/Users/eladsofer/venv/icml/adversarial_updated/data/graph_pkl_files/LADMM_defense_eps_[0.005, 0.025, 0.045, 0.065, 0.085]_NIFGSM.pkl"
    plot_defense_graph(attack_radius_vec, ladmm_nifgsm_path, attack="NIFGSM",
                       algorithm='LADMM', load=False, interpolation=True)

    # Attack graph
    admm_bim = load_object(ladmm_bim_path)["final_results_adv"]["admm"]
    admm_nifgsm = load_object(ladmm_nifgsm_path)["final_results_adv"]["admm"]
    ista_bim = load_object(lista_bim_path)["final_results_adv"]["ista"]
    ista_nifgsm = load_object(lista_nifgsm_path)["final_results_adv"]["ista"]

    vanilla_algo_attacks = {"ADMM-BIM": admm_bim, "ADMM-NIFGSM": admm_nifgsm,
                            "ISTA-BIM": ista_bim, "ISTA-NIFGSM": ista_nifgsm}
    plot_paper_graph(attack_radius_vec, vanilla_algo_attacks, "vanilla_attack.pdf", load=False,interpolation=True)

    # trajectory graph - from LISTA.py








# if __name__ == '__main__':
#     MLSP
#     radius_vec = np.linspace(eps_min, eps_max, r_step)
#     ISTA_min_distances = np.load('data/stack/version1/matrices/ISTA_total_norm.npy')
#     ADMM_min_distances = np.load('data/stack/version1/matrices/ADMM_total_norm.npy')
#     plt.figure()
#     # plt.style.use('plot_style.txt')
#
#     plt.plot(radius_vec, ADMM_min_distances.mean(axis=0), '.-')
#     plt.plot(radius_vec, ISTA_min_distances.mean(axis=0))
#     plt.xlabel(r'$\epsilon$')
#     plt.ylabel(r'${\|\| {s}^{\star} - {s}_{\rm adv}^{\star} \|\|}_2$')
#     plt.legend(['ADMM', 'ISTA'])
#     save_fig('norm2_combined.pdf')
#     plt.show()
#
#     x, s = generate_signal()
#     plt.figure(figsize=(8, 8))
#     plt.subplot(2, 1, 1)
#     plt.plot(x, label='observation')
#     plt.xlabel('Index', fontsize=10)
#     plt.ylabel('Value', fontsize=10)
#     plt.legend()
#     plt.subplot(2, 1, 2)
#
#     plt.plot(s[0], label='sparse signal', color='k')
#     plt.xlabel('Index', fontsize=10)
#     plt.ylabel('Value', fontsize=10)
#     plt.legend()
#     plt.show()