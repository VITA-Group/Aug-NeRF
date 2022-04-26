import os
import time 
import torch
import random
import shutil
import numpy as np  
import torch.nn as nn 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from adv.utils import * 

__all__ = ['attack_pgd']

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

# https://github.com/locuslab/robust_overfitting/blob/master/train_cifar.py
def attack_pgd(model, xs, y, perturb_sizes, epsilons, alphas, loss_fn, attack_iters=1, unadv=False, norm="l_inf", restarts=1):

    if len(perturb_sizes) == 0:
        return {}

    device = y.device

    # convert single argument to dictionary
    if isinstance(attack_iters, int):
        n_iters = attack_iters
        attack_iters = {k: n_iters for k in perturb_sizes.keys()}
    if isinstance(norm, str):
        norm_type = norm
        norm = {k: norm_type for k in perturb_sizes.keys()}

    max_loss, max_delta = {}, {}
    for k, size in perturb_sizes.items():
        max_loss[k] = torch.zeros(y.shape[0], device=device)
        max_delta[k] = torch.zeros(size, device=device)

    for _ in range(restarts):
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
            # delta = clamp(delta, 0-X, 1-X)
            d.requires_grad = True
            delta[k] = d

        max_attack_iters = max(attack_iters.values())
        for i_attack in range(max_attack_iters):
            output = model(*xs, adv_perturb=delta) # add the normalize operation inside model
            loss = loss_fn(output, y)
            if unadv:
                loss = -loss
            loss.backward()

            for k, d0 in delta.items():
                try:
                    # stop updating perturbation after reaching the specified attack iterations
                    if i_attack < attack_iters[k]:
                        g = d0.grad.detach()
                        d = d0.clone()
                        if norm[k] == "l_inf":
                            d = torch.clamp(d + alphas[k] * torch.sign(g), min=-epsilons[k], max=epsilons[k])
                        elif norm[k] == "l_2":
                            ndim = g.ndim
                            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view([-1] + [1]*ndim)
                            scaled_g = g / (g_norm + 1e-10)
                            d = (d + scaled_g * alphas[k]).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilons[k]).view_as(d)

                        d0.data = d

                    # must zero gradients to prevent accumulation
                    d0.grad.zero_()
                except AttributeError as e:
                    print("PGD error at key:", k)
                    raise e

        all_loss = loss_fn(model(*xs, adv_perturb=delta), y, reduction='none')
        for k, d in delta.items():
            if max_delta[k].shape[0] == all_loss.shape[0]: # sample-wise perturb
                sel_mask = (all_loss >= max_loss[k])
                max_delta[k][sel_mask] = d.detach()[sel_mask]
                max_loss[k] = torch.max(max_loss[k], all_loss)
            else: # other perturb
                if torch.mean(max_loss[k]) < torch.mean(all_loss):
                    max_delta[k] = d.detach()
                    max_loss[k] = all_loss

    # ensure all delta are detached
    for k, d in max_delta.items():
        max_delta[k] = d.detach()

    return max_delta

def train_epoch_adv(train_loader, model, criterion, optimizer, epoch, args):
    
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

        # compute output
        output_adv = model(image_adv)
        loss = criterion(output_adv, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

def test_adv(val_loader, model, criterion, args):
    """
    Run adversarial evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    start = time.time()
    for i, (image, target) in enumerate(val_loader):

        image = image.cuda()
        target = target.cuda()

        #adv samples
        delta = attack_pgd(model, image, target, args.test_eps, args.test_alpha, args.test_step, args.test_norm)
        delta.detach()
        image_adv = torch.clamp(image + delta[:image.size(0)], 0, 1)

        # compute output
        with torch.no_grad():
            output = model(image_adv)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('Robust Accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg
