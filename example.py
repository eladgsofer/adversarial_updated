import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn as nn
from scipy.linalg import eigvalsh
import numpy as np
import random

from ista import BIM
import copy

from data import SimulatedData
import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.set_default_dtype(torch.float64)
adv_mode_vec = [-1, 0.001475, 0.0015, 0.001525, 0.00155]
BATCH_SIZE = 10


def ista(x, H, b_s, rho=0.5, L=1, max_itr=300):
    loss_vs_iter = np.zeros(max_itr)
    s_hat = torch.zeros(H.shape[1])
    proj = torch.nn.Softshrink(lambd=rho / L)
    for idx in range(max_itr):
        s_tild = s_hat - 1 / L * (H.T @ (H @ s_hat - x))
        s_hat = proj(s_tild)
        # Aggregate each iteration's MSE loss
        loss_vs_iter[idx] = F.mse_loss(s_hat, b_s, reduction="sum").data.item()

    return loss_vs_iter[-1]


def create_data_set(H, n, m, k, N=1000, batch_size=BATCH_SIZE, signal_dev=0.5, noise_dev=0.01):
    # Initialization
    x = torch.zeros(N, n)
    s = torch.zeros(N, m)

    # Create signals
    for i in range(N):
        # Create a k-sparsed signal s
        index_k = np.random.choice(m, k, replace=False)
        peaks = signal_dev * np.random.randn(k)

        s[i, index_k] = torch.from_numpy(peaks).to(s)

        # X = Hs+w
        x[i, :] = H @ s[i, :] + noise_dev * torch.randn(n)

    simulated = SimulatedData(x=x, H=H, s=s)
    data_loader = Data.DataLoader(dataset=simulated, batch_size=batch_size, shuffle=True)
    return data_loader


N = 1500  # number of samples
n = 150  # dim(x)
m = 200  # dim(s)
k = 4  # k-sparse signal

# Measurement matrix
H = torch.randn(n, m)
H /= torch.norm(H, dim=0)

# Generate datasets
train_loader = create_data_set(H, n=n, m=m, k=k, N=N, batch_size=BATCH_SIZE)
test_loader = create_data_set(H, n=n, m=m, k=k, N=N, batch_size=N)

x_exm, s_exm = test_loader.dataset.__getitem__(5)
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(x_exm, label='observation')
plt.xlabel('Index', fontsize=10)
plt.ylabel('Value', fontsize=10)
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(s_exm, label='sparse signal', color='k')
plt.xlabel('Index', fontsize=10)
plt.ylabel('Value', fontsize=10)
plt.legend()
plt.show()


def validate(model):
    from copy import deepcopy
    model.load_state_dict(torch.load('data/valid/lista_adv_-1.pth'))
    clean_dataloader = torch.load('data/valid/adv_ds_-1.pth')

    vanilla_lista = deepcopy(model)
    robust_lista = deepcopy(model)
    vanilla_lista.eval()
    robust_lista.eval()

    adv_epsilon_vec = list(np.linspace(0.001, 0.002, 4))

    # Choosing different Adversarial examples ratio
    for adv_mode_eps in adv_mode_vec:

        print("####################### adversarial mode: {0} ########################".format(adv_mode_eps))
        robust_lista.load_state_dict(torch.load('lista_adv_{0}.pth'.format(round(adv_mode_eps, 5))))
        adv_dataloader = torch.load('adv_ds_{0}.pth'.format(round(adv_mode_eps, 5)))
        robust_clean_loss = vanilla_clean_loss = 0
        for step, (b_x, b_s) in enumerate(clean_dataloader):
            sr_hat, _ = robust_lista.forward(b_x)
            sv_hat, _ = vanilla_lista.forward(b_x)

            robust_clean_loss += F.mse_loss(sr_hat, b_s, reduction="sum").data.item()
            vanilla_clean_loss += F.mse_loss(sv_hat, b_s, reduction="sum").data.item()

        ista_loss = ista_apply(clean_dataloader, T_ISTA, H)
        vanilla_clean_loss = vanilla_clean_loss / len(clean_dataloader.dataset)
        robust_clean_loss = robust_clean_loss / len(clean_dataloader.dataset)

        print("vanilla_clean_loss {0}".format(vanilla_clean_loss))
        print("robust_clean_loss {0}".format(robust_clean_loss))
        print("ISTA_clean_loss_{0}".format(ista_loss))

        robust_adv_loss = vanilla_adv_loss = 0
        for step, (b_x, b_s) in enumerate(adv_dataloader):
            sr_hat, _ = robust_lista.forward(b_x)
            sv_hat, _ = vanilla_lista.forward(b_x)

            robust_adv_loss += F.mse_loss(sr_hat, b_s, reduction="sum").data.item()
            vanilla_adv_loss += F.mse_loss(sv_hat, b_s, reduction="sum").data.item()

        ista_loss = ista_apply(adv_dataloader, T_ISTA, H)

        vanilla_adv_loss = vanilla_adv_loss / len(adv_dataloader.dataset)
        robust_adv_loss = robust_adv_loss / len(adv_dataloader.dataset)

        print("vanilla_adv_loss {0}".format(vanilla_adv_loss))
        print("robust_adv_loss {0}".format(robust_adv_loss))
        print("ISTA_adv_loss_{0}".format(ista_loss))


def train(original_model, train_loader, valid_loader, num_epochs=40, attack_radius=0.05):
    """Train a network.
    Returns:
        loss_test {numpy} -- loss function values on test set
    """
    # TODO - Make A,B parameters in the LISTA.

    adv_epsilon_vec = list(np.linspace(0.001, 0.002, 4))
    adv_epsilon_vec.insert(0, -1)
    # Choosing different Adversarial examples ratio

    # Initialization
    model = copy.deepcopy(original_model)
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-05, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    loss_train = np.zeros((num_epochs,))
    loss_test = np.zeros((num_epochs,))

    # Main loop
    adv_train_ds = []
    model.train()
    adv_b = 0
    adv_mode_eps = 0.32

    if adv_mode_eps > 0:
        # Create Adversarial examples
        for step, (b_x, b_s) in enumerate(train_loader):
            if random.random() <= adv_mode_eps:
                print("Adversarial batch {0} was added".format(adv_b))
                adv_b += 1
                adv_x, delta = BIM(model, b_x.detach(), b_s.detach(), eps=attack_radius)
                adv_train_ds.append((adv_x, b_s))
                model.zero_grad()

        # Create new DataLoader
        mix_dataset = copy.deepcopy(train_loader.dataset)
        clean_dataset = copy.deepcopy(train_loader.dataset)
        adv_dataset = copy.deepcopy(train_loader.dataset)

        for (adv_x, s_gt) in adv_train_ds:
            mix_dataset.x = torch.cat((mix_dataset.x, adv_x), 0)
            mix_dataset.s = torch.cat((mix_dataset.s, s_gt), 0)

        train_loader = Data.DataLoader(dataset=mix_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("##### Adversarial epsilon {0} #####".format(adv_mode_eps))
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for step, (b_x, b_s) in enumerate(train_loader):
            s_hat, _ = model.forward(b_x)
            loss = F.mse_loss(s_hat, b_s, reduction="sum")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            train_loss += loss.data.item()

        loss_train[epoch] = train_loss / len(train_loader.dataset)
        scheduler.step()

        # Stop following gradient
        model.eval()

        # Validation upon clean examples
        test_loss = 0
        adv_validation_ds = []
        for step, (b_x, b_s) in enumerate(valid_loader):
            s_hat, _ = model.forward(b_x)
            test_loss += F.mse_loss(s_hat, b_s, reduction="sum").data.item()

        loss_test[epoch] = test_loss / len(valid_loader.dataset)
        # Print
        if epoch % 5 == 0:
            print("Epoch %d, Train loss %.8f, Validation loss %.8f" % (epoch, loss_train[epoch], loss_test[epoch]))

    model.eval()

    if adv_mode_eps > 0:
        # Creating adversarial dataset
        adv_x_ds = torch.tensor([])
        adv_s_ds = torch.tensor([])
        for step, (b_x, b_s) in enumerate(valid_loader):
            s_hat, _ = model.forward(b_x)
            adv_x, delta = BIM(model, b_x.detach(), b_s.detach(), eps=attack_radius)
            adv_x_ds = torch.cat((adv_x_ds, adv_x), 0)
            adv_s_ds = torch.cat((adv_s_ds, b_s), 0)

        adv_val_ds = copy.deepcopy(mix_dataset)
        adv_val_ds.x = adv_x_ds
        adv_val_ds.s = adv_s_ds
        adv_val_loader = Data.DataLoader(dataset=adv_val_ds, batch_size=BATCH_SIZE, shuffle=True)

        # torch.save(adv_val_loader, 'adv_ds_{0}.pth'.format(round(adv_mode_eps, 5)))
        torch.save(adv_val_loader, 'adv_ds_{0}.pth'.format(round(adv_mode_eps, 5)))

    else:
        #
        # torch.save(valid_loader, 'adv_ds_{0}.pth'.format(adv_mode_eps, round(adv_mode_eps, 5)))
        torch.save(train_loader, 'adv_ds_{0}.pth'.format(adv_mode_eps, round(adv_mode_eps, 5)))

    torch.save(model.state_dict(), 'lista_adv_{0}.pth'.format(round(adv_mode_eps, 5)))

        # from copy import deepcopy
        # model.load_state_dict(torch.load('lista_adv_-1.pth'))
        # vanilla_lista = deepcopy(model)
        # model.load_state_dict(torch.load('lista_adv_0.005.pth'))
        # adv_lista = deepcopy(model)
        #
        # adv_x, delta = BIM(vanilla_lista, b_x.detach(), b_s.detach(), eps=attack_radius)
        #
        # vanilla_s_adv, _ = vanilla_lista.forward(adv_x)
        # robust_s_adv, _ = adv_lista.forward(adv_x)
        # #
        # vanilla_loss = F.mse_loss(vanilla_s_adv, b_s, reduction="sum")
        # robust_s_loss = F.mse_loss(robust_s_adv, b_s, reduction="sum")
        #
        # print("vanilla loss {0}".format(vanilla_loss))
        # print("robust loss {0}".format(robust_s_loss))


class LISTA_Model(nn.Module):
    T_LISTA = 5

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

    def forward(self, x, s_gt=None):
        mse_vs_itr = []

        s_hat = self._shrink(self.mu[0, :, :] * self.A(x), self.beta[0, :, :])
        for i in range(1, self.T + 1):
            s_hat = self._shrink(s_hat - self.mu[i, :, :] * self.B(s_hat) + self.mu[i, :, :] * self.A(x),
                                 self.beta[i, :, :], )

            # Aggregate each iteration's MSE loss
            if s_gt is not None:
                mse_vs_itr.append(F.mse_loss(s_hat.detach(), s_gt.detach(), reduction="sum").data.item())

        return s_hat, mse_vs_itr

    @classmethod
    def create_lista_model(cls, H=H):
        # Is there randomness at creating each time new instance?
        n = H.shape[1]
        m = H.shape[1]
        return cls(n=n, m=m, T=cls.T_LISTA, H=H)


def lista_apply(train_loader, test_loader, T, H):
    lista = LISTA_Model.create_lista_model()
    train(lista, train_loader, test_loader)
    validate(lista)



    # # Extract all samples and calculate MSE for each iteration
    # s_gt, x = test_loader.dataset.s, test_loader.dataset.x
    # _, mse_vs_iter = lista(x, s_gt=s_gt)
    #
    # return np.array(mse_vs_iter) / len(test_loader.dataset)


def ista_apply(test_loader, T, H, rho=0.5):
    H = H.cpu()
    m = H.shape[1]
    L = float(eigvalsh(H.t() @ H, eigvals=(m - 1, m - 1)))

    # Aggregate T iterations' MSE loss
    losses = np.zeros((len(test_loader.dataset)))
    loss = []

    for idx, (x, b_s) in enumerate(test_loader.dataset):
        loss.append(ista(x=x, H=H, b_s=b_s, rho=rho, L=L, max_itr=T))

    loss = np.array(loss)

    return loss.mean()


# number of unfolded iteartions

T_LISTA = LISTA_Model.T_LISTA
T_ISTA = 100 * T_LISTA

# TODO play with the batch_size to reduce/increase adversarial examples. determine 1000=dataset.
# Train and apply LISTA with T iterations / layers
lista_mse_vs_iter = lista_apply(train_loader, test_loader, T_LISTA, H)

# ista_mse_vs_iter = ista_apply(test_loader, T_ISTA, H)

# plot the resutls
# fig = plt.figure()
# plt.plot(range(T_ISTA), ista_mse_vs_iter, label='ISTA', color='b', linewidth=0.5)
# plt.plot(range(T_LISTA), lista_mse_vs_iter, label='LISTA', color='r', linewidth=2)
# plt.xlabel('Number of iterations', fontsize=10)
# plt.ylabel('MSE', fontsize=10)
# plt.yscale("log")
# plt.legend()
# plt.show()
