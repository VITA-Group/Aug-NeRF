
# https://github.com/csdongxian/AWP/blob/main/AT_AWP/awp.py

import os
import time 
import copy 
import torch
import random
import shutil
import numpy as np  
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
import torch.backends.cudnn as cudnn

from adv.pgd import attack_pgd
from adv.utils import * 

__all__ = ['AdvWeightPerturb']

EPS = 1E-20

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict

def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])

# https://github.com/locuslab/robust_overfitting/blob/master/train_cifar.py
# def attack_pgd(model, X, y, epsilon, alpha, attack_iters, norm="l_inf", 
#                 early_stop=False, restarts=1):
#     max_loss = torch.zeros(y.shape[0]).cuda()
#     max_delta = torch.zeros_like(X).cuda()
#     for _ in range(restarts):
#         delta = torch.zeros_like(X).cuda()
#         if norm == "l_inf":
#             delta.uniform_(-epsilon, epsilon)
#         elif norm == "l_2":
#             delta.normal_()
#             d_flat = delta.view(delta.size(0),-1)
#             n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
#             r = torch.zeros_like(n).uniform_(0, 1)
#             delta *= r/n*epsilon
#         else:
#             raise ValueError
#         delta = clamp(delta, 0-X, 1-X)
#         delta.requires_grad = True
#         for _ in range(attack_iters):
#             output = model(X + delta) # add the normalize operation inside model
#             if early_stop:
#                 index = torch.where(output.max(1)[1] == y)[0]
#             else:
#                 index = slice(None,None,None)
#             if not isinstance(index, slice) and len(index) == 0:
#                 break

#             loss = F.cross_entropy(output, y)
#             loss.backward()
#             grad = delta.grad.detach()
#             d = delta[index, :, :, :]
#             g = grad[index, :, :, :]
#             x = X[index, :, :, :]
#             if norm == "l_inf":
#                 d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
#             elif norm == "l_2":
#                 g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
#                 scaled_g = g/(g_norm + 1e-10)
#                 d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
#             d = clamp(d, 0 - x, 1 - x)
#             delta.data[index, :, :, :] = d
#             delta.grad.zero_()

#         all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
#         max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
#         max_loss = torch.max(max_loss, all_loss)
#     return max_delta

class AdvWeightPerturb(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs, targets, delta, loss_fn, unadv=False):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        
        loss = -loss_fn(self.proxy(*inputs, adv_perturb=delta), targets)
        if unadv:
            loss = -loss

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)

def train_epoch_adv_awp(train_loader, model, awp_adversary, criterion, optimizer, epoch, args):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    start = time.time()
    for i, (image, target) in enumerate(train_loader):

        image = image.cuda()
        target = target.cuda()

        #adv samples
        model.eval() # https://arxiv.org/pdf/2010.00467.pdf
        delta = attack_pgd(model, image, target, args.train_eps, args.train_alpha, args.train_step, args.train_norm)
        delta.detach()
        image_adv = torch.clamp(image + delta[:image.size(0)], 0, 1)
        model.train()

        if epoch >= args.awp_warmup:
            awp = awp_adversary.calc_awp(inputs_adv=image_adv, targets=target)
            awp_adversary.perturb(awp)
        # compute output
        output_adv = model(image_adv)
        loss = criterion(output_adv, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= args.awp_warmup:
            awp_adversary.restore(awp)

        output = output_adv.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('adversarial train accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg
