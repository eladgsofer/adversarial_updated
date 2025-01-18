__author__ = 'Elad Sofer <elad.g.sofer@gmail.com>'

import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import seaborn as sns
import time
from utills import save_fig

from visualize_model import LandscapeWrapper


sns.set()
np.random.seed(0)

# AccAltProj configuration

n = 2500
m = n
r = 5
alpha = 0.4
c = 1


def BIM(model, x, L_gt, S_gt, eps=0.1, alpha=0.1, steps=3):
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
    x = x.clone()#.to(device)
    L_gt, S_gt = L_gt.clone(), S_gt.clone()
    # s_gt = s_gt.clone().to(device)

    # loss = nn.MSELoss()

    original_x = x.data
    original_x.requires_grad = False

    adv_x = x.clone().detach()

    for step in range(steps):

        print("#################### BIM Step {0} distance ratio: {1} ####################".format(step, ((adv_x - original_x).norm() / original_x.norm()).item()))

        adv_x.requires_grad = True
        l_hat, s_hat, errs = model(adv_x)
        model.zero_grad()

        # Calculate loss
        cost = torch.linalg.norm(original_x - l_hat - s_hat, 'fro')
        print("#################### COST: {0} ####################".format(cost.item()/torch.linalg.norm(original_x, 'fro')))

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

    return adv_x, delta  # grad is the gradient (perturbation)


# ISTA
class AccAltProj(nn.Module, LandscapeWrapper):
    """
       Implements the Iterative Shrinkage-Thresholding Algorithm (ISTA) for sparse signal recovery.
       Args:
           H (torch.Tensor): Sensing matrix.
           mu (float): Solver parameter for gradient descent step.
           rho (float): Regularization parameter for L1-norm penalty.
           max_iter (int): Maximum number of iterations.
           eps (float): Convergence threshold.

       Attributes:
           H (torch.Tensor): Sensing matrix.
           rho (float): Regularization parameter for L1-norm penalty.
           mu (float): Solver parameter for gradient descent step.
           max_iter (int): Maximum number of iterations.
           eps (float): Convergence threshold.
           s (torch.Tensor): Initial estimate of the sparse signal.
           model_params (nn.Parameter): Model parameters used for visualization.
       """
    def __init__(self, beta=None, beta_init=None, max_iter=100, tol=1e-3, gamma=0.7, mu=5):
        super(AccAltProj, self).__init__()

        self.mu = torch.tensor(mu)
        self.gamma = torch.tensor(gamma)
        if beta is None:
            self.beta = 1 / (2 * np.power(m * n, 1 / 4))
        if beta_init is None:
            self.beta_init = 4 * self.beta

        self.max_iter = max_iter
        self.tol = tol

        # initial estimate
        self.s = None
        self.model_params = None

    @staticmethod
    def wthresh(x, beta):
        """
        Applies the shrinkage operator to the input tensor 'x' with a threshold of 'beta'.
        :param x: Input tensor.
        :param beta: Threshold value.
        :return: Resulting tensor after applying shrinkage.
        """
        # Shrinking towards 0 by Beta parameter.
        return torch.mul(torch.sign(x), torch.max(torch.abs(x) - beta, torch.tensor(0)))

    def forward(self, D):
        """
        Performs ISTA reconstruction on the input signal 'x'.
        :param x: Input signal to reconstruct. (torch.Tensor)
        :return  torch.Tensor: Reconstructed sparse signal.
        :return list: List of recovery errors at each iteration.
        """

        m, n = D.shape

        norm_of_D = np.linalg.norm(D.detach().numpy(), 'fro')

        err = -1 * np.ones(self.max_iter)
        timer = -1 * np.ones(self.max_iter)

        tic = time.time()
        # first lambda isn't the same like svds
        zeta = self.beta_init * torch.linalg.svd(D, True)[1][0]

        S = self.wthresh(D, zeta)

        U, Sigma, V = torch.linalg.svd(D - S)
        U = torch.fliplr(U)[:, -r:]
        V = torch.fliplr(V.T)[:, -r:]
        Sigma = torch.flip(Sigma[:r], [0])
        L = U @ torch.diag(Sigma) @ V.T

        zeta = self.beta * Sigma[0]
        S = self.wthresh(D - L, zeta)

        init_timer = time.time() - tic
        init_err = np.linalg.norm(D.detach().numpy() - L.detach().numpy() - S.detach().numpy(), 'fro') / norm_of_D

        print(f'Init: error: {init_err}, timer: {init_timer}')

        for t in range(self.max_iter):
            tic = time.time()

            # Update L
            Z = D - S
            Q1, R1 = torch.linalg.qr(Z.T @ U - V @ (((Z @ V).T) @ U))
            Q2, R2 = torch.linalg.qr(Z @ V - U @ (U.T @ Z @ V))
            # Does QR is unique?

            M = torch.hstack([torch.vstack([U.T @ Z @ V, R1.T]),
                               torch.vstack([R2, torch.zeros_like(R2)])])

            U_of_M, Sigma, V_of_M = torch.linalg.svd(M, full_matrices=False)

            U = torch.hstack([U, Q2]) @ torch.fliplr(U_of_M)[:, -r:]
            V = torch.hstack([V, Q1]) @ torch.fliplr(V_of_M.T)[:, -r:]
            L = U @ torch.diag(torch.flip(Sigma[:r], [0])) @ V.T

            # Update S
            zeta = self.beta * (Sigma[r] + torch.pow(self.gamma, t) * Sigma[0])
            S = self.wthresh(D - L, zeta)

            # Stop Condition
            err[t] = np.linalg.norm(np.array(D.detach().numpy() - L.detach().numpy() - S.detach().numpy()), 'fro') / norm_of_D
            timer[t] = time.time() - tic
            if err[t] < self.tol:
                print(f'Total {t + 1} iteration, final error: {err[t]}, '
                      f'total time without init: {np.sum(timer[timer > 0])}, '
                      f'with init: {np.sum(timer[timer > 0]) + init_timer}')
                return L, S, err

            print(f'Iteration {t + 1}: error: {err[t]}, timer: {timer[t]}')

        print(f'Maximum iterations reached, final error: {err[-1]}')
        return L, S, err
        #
        # #### ISTA
        #     s_prev = self.s
        #     # proximal gradient step
        #     temp = torch.matmul(self.H, s_prev) - x
        #
        #     g_grad = s_prev - torch.mul(self.mu, torch.matmul(self.H.T, temp))
        #     self.s = self.shrinkage(g_grad, np.multiply(self.mu, self.rho))
        #
        #     # cease if convergence achieved
        #     if torch.sum(torch.abs(self.s - s_prev)).item() <= self.eps:
        #         break
        #
        #     # save recovery error
        #     error = self.loss_func(self.s, x)
        #     recovery_errors.append(error)
        #
        # return self.s, recovery_errors

    def set_model_visualization_params(self):
        """
        Sets the model parameters for visualization for the visualize_model module to operate.
        """
        self.model_params = nn.Parameter(self.s.detach(), requires_grad=False)

    def loss_func(self, s, x_sig):
        """
        Computes the loss function given the estimated sparse signal 's' and its observation 'x_sig'.
        :param s: Estimated sparse signal.
        :param x_sig: observation signal x = Hs + w, where w is a Gaussian noise.
        :return: Loss value.
        """
        return 0.5 * torch.sum((torch.matmul(self.H, s) - x_sig) ** 2).item() + self.rho * s.norm(p=1).item()

    @staticmethod
    def copy(other):
        """
        Creates a deep copy of the 'other' object.
        Args: other (ISTA): ISTA object to copy.
        Returns: ISTA: Deep copy of the 'other' object.
        """
        return copy.deepcopy(other)

    @classmethod
    def create_AccAltProj(cls, beta=None, beta_init=None, max_iter=100, tol=1e-3, gamma=0.7, mu=5):
        """
        Creates an instance of the ISTA class with the specified parameters.
        :param H: Sensing matrix.
        :param step_size: Solver parameter for gradient descent step.
        :param rho: Regularization parameter for L1-norm penalty.
        :param max_iter: Maximum number of iterations.
        :param eps_threshold: Convergence threshold.
        :return: ISTA object.
        """
        return cls(beta=beta, beta_init=beta_init, max_iter=max_iter, tol=tol, gamma=gamma, mu=mu)

def generate_matrices(matrice_amount=100):
    matrices = []
    for i in range(matrice_amount):
        # Generate a RPCA problem
        A_generater = np.random.randn(m, r)
        B_generater = np.random.randn(r, n)
        L_true = np.dot(A_generater, B_generater)
        # norm_of_L_true = np.linalg.norm(L_true, 'fro')

        S_supp_idx = np.random.choice(m * n, size=int(round(alpha * m * n)), replace=False)
        S_range = c * np.mean(np.abs(L_true))
        S_temp = 2 * S_range * np.random.rand(m, n) - S_range
        S_true = np.zeros((m, n))
        S_true.flat[S_supp_idx] = S_temp.flat[S_supp_idx]
        # norm_of_S_true = np.linalg.norm(S_true, 'fro')

        D = L_true + S_true
        matrices.append((torch.tensor(D), torch.tensor(S_true), torch.tensor(L_true)))
    return matrices
def execute():
    """
    Perform a series of operations on generated signals:
    1. Generate 'c' (set to 100) signals of the form x_i = Hs + w, where w follows a Gaussian distribution.
    2. Perform ISTA reconstruction on each signal x to obtain s^*.
    3. Perform BIM adversarial attack with different epsilon values to obtain x_{adv}.
    4. Perform ISTA reconstruction on each signal x_{adv} to obtain s_{adv}.
    5. Aggregate the L2 norm ||s^* - s^*_{adv}|| for each signal and epsilon value.
    6. Plot the loss surfaces in various forms (3D, 2D, 1D) and other related graphs.
    """

    matrices_N = 100
    radius_n = 5

    matrices = generate_matrices(matrices_N)
    ##########################################################

    radius_vec = np.linspace(0.0001, 0.0005, radius_n)
    #radius_vec = [0.1, 0.001]

    attack_ratios_hist = dict.fromkeys(radius_vec,0)
    adv_loss_hist = dict.fromkeys(radius_vec,0)
    gt_loss = 0

    # TODO think how to measure performance..
    for mat_idx, (D_original, S_original, L_original) in enumerate(matrices):
        # ISTA without an attack reconstruction

        AccAltProj_t_model = AccAltProj.create_AccAltProj()
        l_hat, s_hat, err_gt = AccAltProj_t_model.forward(D_original.detach())
        print("#### RPCA decomposition {0} convergence: iterations: {1} ####".format(mat_idx, len(err_gt)))
        # L_gt, S_gt = L_gt.detach(), S_gt.detach()
        gt_loss += (D_original - l_hat - s_hat).norm('fro').item()
        for e_idx, attack_eps in enumerate(radius_vec):
            print("Performing BIM to get Adversarial Perturbation - epsilon: {0}".format(attack_eps))
            AccAltProj_adv_model = AccAltProj.create_AccAltProj()

            adv_D, _ = BIM(AccAltProj_adv_model, D_original, S_original, L_original,
                           alpha=0.01, eps=attack_eps, steps=4)
            adv_D = adv_D.detach()

            L_adv, S_adv, _ = AccAltProj_adv_model(adv_D)

            adv_loss_hist[attack_eps] += (torch.linalg.norm(D_original - L_adv - S_adv, 'fro')/torch.linalg.norm(D_original, 'fro')).item()

            attack_ratios_hist[attack_eps] += ((adv_D - D_original).norm(2) / D_original.norm(2)).item()

    gt_loss/=matrices_N

    print("ground-truth loss is {0}".format(gt_loss))


    # with open('filename.pickle', 'rb') as handle:
    #     b = pickle.load(handle)

    loss_hist = {eps: total_loss/matrices_N for eps, total_loss in adv_loss_hist.items()}
    attack_ratios_hist = {eps: ratio / matrices_N for eps, ratio in attack_ratios_hist.items()}

    with open('rpca_loss_hist_{0}.pickle'.format(attack_eps), 'wb') as handle:
        pickle.dump(loss_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('attack_ratios_hist_{0}.pickle'.format(attack_eps), 'wb') as handle:
        pickle.dump(loss_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.figure()
    plt.plot(loss_hist.keys(), loss_hist.values())
    plt.xlabel('epsilon')
    plt.ylabel('Loss = ||D-L_adv-S_adv||/||D||')
    save_fig('loss_rpca.pdf')
    plt.show()

    plt.figure()
    plt.plot(attack_ratios_hist.keys(), attack_ratios_hist.values())
    plt.xlabel('epsilon')
    plt.ylabel('ratio')
    save_fig('ratio_rpca_{0}.pdf')
    plt.show()



if __name__ == '__main__':
    execute()