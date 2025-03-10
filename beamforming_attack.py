__author__ = 'Elad Sofer <elad.g.sofer@gmail.com>'

import numpy as np
import torch
import matplotlib.pyplot as plt

# official beam_forming module in - https://github.com/ortalagiv/learn-to-rapidly-optimize-hybrid-precoding
# for more info about beam-forming - https://arxiv.org/abs/2301.00369
from beam_forming import N, L, B, num_of_iter_pga, ProjGA, H_test
from utills import device
import os
from utills import save_object, load_object

def beamforming_NIFGSM(model, h, eps=0.1, alpha=1,decay=1, steps=30,randomize=True):
    """
    Performs a variant of the Basic Iterative Method (BIM) attack on the beam-forming model.
    :param model: The beam-forming model.
    :param h: input matrix H.
    :param eps: The epsilon range to clip the change between adversarial and original images. (default: 0.1)
    :param alpha: The scaling factor for the gradient. (default: 1)
    :param steps: The number of BIM attack steps. (default: 30)
    :return: A tuple containing the adversarial tensor, the pertubation, delta,
             and the predicted outputs wa_hat and wd_hat.
    """

    h = h.clone().to(device)
    momentum = torch.zeros_like(h).detach()

    original_x = h.data
    adv_h = h.clone().detach()

    if randomize:
        delta = torch.rand_like(adv_h, requires_grad=False)
        delta.data = delta.data * 2 * eps - eps
        delta.data = delta.clamp(-eps, eps)
    else:
        delta = torch.zeros_like(adv_h, requires_grad=False)

    adv_h = adv_h + delta

    for step in range(steps):
        # print("BIM Step {0}".format(step))
        adv_h.requires_grad = True
        model.zero_grad()
        _, wa_hat, wd_hat = model(h=adv_h)

        R = model.objec(h=original_x, wa=wa_hat, wd=wd_hat)

        grad = torch.autograd.grad(R, adv_h)[0]
        grad = decay * momentum + grad
        momentum = grad

        # Grad is calculated
        delta = alpha * grad

        # Stop following gradient changes
        adv_h = adv_h.clone().detach()

        adv_h = adv_h - delta

        # Clip the change between the adverserial images and the original images to an epsilon range
        real_eta = torch.clamp((adv_h - original_x).real, min=-eps, max=eps)
        imag_eta = torch.clamp((adv_h - original_x).imag, min=-eps, max=eps)

        adv_h = original_x + torch.complex(real_eta, imag_eta)

    return adv_h, delta, wa_hat, wd_hat

def beamforming_BIM(model, h, eps=0.1, alpha=1, steps=30):
    """
    Performs a variant of the Basic Iterative Method (BIM) attack on the beam-forming model.
    :param model: The beam-forming model.
    :param h: input matrix H.
    :param eps: The epsilon range to clip the change between adversarial and original images. (default: 0.1)
    :param alpha: The scaling factor for the gradient. (default: 1)
    :param steps: The number of BIM attack steps. (default: 30)
    :return: A tuple containing the adversarial tensor, the pertubation, delta,
             and the predicted outputs wa_hat and wd_hat.
    """

    h = h.clone().to(device)

    original_x = h.data
    adv_h = h.clone().detach()

    for step in range(steps):
        # print("BIM Step {0}".format(step))
        adv_h.requires_grad = True
        model.zero_grad()
        _, wa_hat, wd_hat = model(h=adv_h)

        R = model.objec(h=original_x, wa=wa_hat, wd=wd_hat)

        grad = torch.autograd.grad(R, adv_h)[0]

        # Grad is calculated
        delta = alpha * grad

        # Stop following gradient changes
        adv_h = adv_h.clone().detach()

        adv_h = adv_h - delta

        # Clip the change between the adverserial images and the original images to an epsilon range
        real_eta = torch.clamp((adv_h - original_x).real, min=-eps, max=eps)
        imag_eta = torch.clamp((adv_h - original_x).imag, min=-eps, max=eps)

        adv_h = original_x + torch.complex(real_eta, imag_eta)

    return adv_h, delta, wa_hat, wd_hat


##########################################################

def execute(attack):
    """
    Executes a BIM attack on the beam-forming algorithm using different epsilon values.

    This function performs the following steps:
    1. Iterates over the dataset for each H matrix and performs the following:
       a. Creates a new instance of the ProjGA model with the given mu.
       b. Performs a BIM adversarial attack with different epsilon values on each H matrix.
       c. Computes the achievable rate.
       d. Stores the rate in the rates array at index (h_idx, e_idx).

    2. Plots a figure showing the attack radius against the mean achievable rate for all H matrices.
    """

    mu = torch.tensor([[50 * 1e-2] * (B + 1)] * num_of_iter_pga, requires_grad=False)

    classical_model = ProjGA(mu)
    sum_rate_class, wa, wd = classical_model.forward(H_test, N, L, B, num_of_iter_pga)

    wa_original, wd_original, original_h = wa.detach(), wd.detach(), H_test.detach()
    print("BeamForming Rate (Un-attacked): {0}".format(
        classical_model.objec(h=original_h, wa=wa_original, wd=wd_original).mean().norm(2).item()))

    # present noise scalar which yields 3.6 rate (benchmarking)
    noise_scalar = 0.81030
    print("BeamForming Rate (attacked via adding traditional noise with ratio) "
          "rate: {0} ratio: {1}".format(classical_model.objec(h=noise_scalar * original_h, wa=wa_original,
                                                              wd=wd_original).mean().item(), noise_scalar))

    attack_radius = np.linspace(0.002, 0.2, 10)
    mu = torch.tensor([[50 * 1e-2] * (B + 1)] * num_of_iter_pga, requires_grad=False)
    rates = np.zeros(((H_test.shape[1]), len(attack_radius)))

    for h_idx in range(H_test.shape[1]):
        print(f"#### iteration {h_idx} ####")
        original_h = H_test[:, h_idx, :, :].reshape((16, 1, 4, 12)).detach()

        for e_idx, eps in enumerate(attack_radius):
            bf_model = ProjGA(mu)
            if attack=="BIM":
                _, _, wa_hat, wd_hat = beamforming_BIM(bf_model, original_h, eps=eps)
            else:
                _, _, wa_hat, wd_hat = beamforming_NIFGSM(bf_model, original_h,eps=eps)

            attacked_rate = classical_model.objec(h=original_h, wa=wa_hat, wd=wd_hat).norm(2).item()

            rates[h_idx, e_idx] = attacked_rate

    y = rates.mean(axis=0)
    save_object({'epsilon': attack_radius, 'Achievable Rate': y}, f'beamforming_{attack}.pdf')
    plt.figure()
    plt.plot(attack_radius, y)
    plt.xlabel('$\epsilon$')
    plt.ylabel('Achievable Rate')
    plt.show()


if __name__ == '__main__':
    for attack in ["BIM", "NIFGSM"]:
        execute(attack)
