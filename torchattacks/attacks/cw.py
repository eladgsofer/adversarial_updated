import torch
import torch.nn as nn
import torch.optim as optim

from ..attack import Attack


class CW(Attack):
    r"""
    CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        c (float): c in the paper. parameter for box-constraint. (Default: 1)    
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (Default: 50)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)

    .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01)
        >>> adv_images = attack(images, labels)

    .. note:: Binary search for c is NOT IMPLEMENTED methods in the paper due to time consuming.

    """

    def __init__(self, model, c=1, steps=5, lr=0.05):
        super().__init__("CW", model)
        self.c = c
        self.steps = steps
        self.lr = lr
        self.supported_mode = ["default", "targeted"]

    def normalize(self, x, y):
        scale_factor_x, scale_factor_y = abs(x).max(dim=1, keepdim=True).values, abs(y).max(dim=1, keepdim=True).values
        x = x / (2*scale_factor_x) + 0.5
        y = y / (2*scale_factor_y) + 0.5
        return x, y, scale_factor_x

    def inverse_normalize(self, x, scale_factor_x):
        return (x-0.5) * 2 * scale_factor_x

    def forward(self, x, y, **kwargs):
        r"""
        Overridden.
        """
        x, y, scale_fac_x = self.normalize(x, y)
        for k, v in kwargs.items():
            setattr(self, k, v)

        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(x=x).detach()
        w.requires_grad = True

        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            # Get adversarial images
            adv_x = self.tanh_space(w)

            # Calculate loss
            current_L_inf = abs(adv_x - x).max(axis=1)[0]
            L_inf = current_L_inf.sum()

            outputs, _ = self.get_logits(adv_x)
            f_loss = self.f(outputs, y).sum()

            cost = L_inf + self.c * f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        return self.inverse_normalize(adv_x.detach(), scale_fac_x), None

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        # atanh is defined in the range -1 to 1
        # 2delta tanw+1-2x
        #tanw = 2delta+2x-1
        #
        return self.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    # f-function in the paper
    def f(self, outputs, labels):
        loss = nn.MSELoss()
        if outputs.shape!=labels.shape:
            cost = loss(outputs, labels.T)
        else:
            cost = loss(outputs, labels)

        if self.targeted:
            return cost
        else:
            return -1*cost
