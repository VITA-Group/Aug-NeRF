import os
import time 
import torch
import random
import shutil
import numpy as np  
import torch.nn as nn 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

__all__ = ['attack_random']

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

# https://github.com/locuslab/robust_overfitting/blob/master/train_cifar.py
def attack_random(model, xs, y, perturb_sizes, epsilons, norm="l_inf", restarts=1):

    if len(perturb_sizes) == 0:
        return {}

    device = y.device

    if isinstance(norm, str):
        norm_type = norm
        norm = {k: norm_type for k in perturb_sizes.keys()}

    max_loss, max_delta = {}, {}
    for k, size in perturb_sizes.items():
        max_loss[k] = torch.zeros(y.shape[0], device=device)
        max_delta[k] = torch.zeros(size, device=device)

    delta = {}
    for k, size in perturb_sizes.items():
        d = torch.zeros(size, device=device)
        if norm[k] == "l_inf":
            d.uniform_(-epsilons[k], epsilons[k])
        elif norm[k] == "l_2":
            d.normal_()
            ndim = d.ndim
            d_flat = d.view(d.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view([d.size(0)] + [1]*(ndim-1))
            r = torch.zeros_like(n).uniform_(0, 1)
            d *= r / n * epsilons[k]
        else:
            raise ValueError

        d.requires_grad = False
        delta[k] = d

    return delta
