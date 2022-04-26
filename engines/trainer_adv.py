import os, sys
import math, time, random

import numpy as np

import imageio
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.image import to8b, img2mse, mse2psnr
from engines.trainer import save_checkpoint
from engines.eval import eval_one_view, evaluate, render_video
from adv.pgd import attack_pgd
from adv.rand import attack_random
from adv.awp import AdvWeightPerturb

def broadcast_lists(*lists):
    lengths = [len(l) for l in lists]
    max_len = max(lengths)
    ret_lists = []
    for l in lists:
        if len(l) == max_len:
            ret_lists.append(l)
        elif len(l) == 1:
            ret_lists.append([l[0]] * max_len)
        else:
            raise ValueError(f'Unable to broadcast a list of length {len(l)} to {max_len}')
    return ret_lists

def get_adv_templates(adv, H, W, near, far, num_images, num_rays, args):

    perturb_sizes, epsilons, alphas = {}, {}, {}
    iters, norms = {}, {}
    
    broadcast = broadcast_lists(args.adv, args.pgd_alpha, args.pgd_eps, args.pgd_iters, args.pgd_norm)
    for adv, pgd_alpha, pgd_eps, pgd_iters, pgd_norm in zip(*broadcast):
        iters[adv] = pgd_iters
        norms[adv] = pgd_norm

        if adv  == 'zval_c':
            perturb_sizes['zval_c'] = [num_rays, args.N_samples]
            epsilons['zval_c'] = (far-near) / (args.N_samples-1) / 2. * pgd_eps
            alphas['zval_c'] = pgd_alpha
        elif adv == 'pts_c':
            perturb_sizes['pts_c'] = [num_rays, args.N_samples, 3]
            epsilons['pts_c'] = min(2. / max(H, W), (far-near) / (args.N_samples-1) / 2.) * pgd_eps
            alphas['pts_c'] = pgd_alpha

        elif adv == 'zval_f':
            perturb_sizes['zval_f'] = [num_rays, args.N_samples+args.N_importance]
            epsilons['zval_f'] = (far-near) / (args.N_samples+args.N_importance-1) / 2. * pgd_eps
            alphas['zval_f'] = pgd_alpha
        elif adv == 'pts_f':
            perturb_sizes['pts_f'] = [num_rays, args.N_samples+args.N_importance, 3]
            epsilons['pts_f'] = min(2. / max(H, W), (far-near) / (args.N_samples+args.N_importance-1) / 2.) * pgd_eps
            alphas['pts_f'] = pgd_alpha

        elif adv == 'raw_c':
            perturb_sizes['raw_c'] = [num_rays, args.N_samples, 4]
            epsilons['raw_c'] = pgd_eps
            alphas['raw_c'] = pgd_alpha
        elif adv == 'raw_f':
            perturb_sizes['raw_f'] = [num_rays, args.N_samples+args.N_importance, 4]
            epsilons['raw_f'] = pgd_eps
            alphas['raw_f'] = pgd_alpha

        elif adv == 'feat_c':
            perturb_sizes['feat_c'] = [num_rays, args.N_samples, args.netwidth]
            epsilons['feat_c'] = pgd_eps
            alphas['feat_c'] = pgd_alpha
        elif adv == 'feat_f':
            perturb_sizes['feat_f'] = [num_rays, args.N_samples+args.N_importance, args.netwidth_fine]
            epsilons['feat_f'] = pgd_eps
            alphas['feat_f'] = pgd_alpha

        elif adv == 'cam_r':
            perturb_sizes['cam_r'] = [num_cameras, 3]
            epsilons['cam_r'] = np.deg2rad(pgd_eps)
            alphas['cam_r'] = pgd_alpha
        elif adv == 'cam_t':
            perturb_sizes['cam_t'] = [num_cameras, 3]
            epsilons['cam_t'] = pgd_eps
            alphas['cam_t'] = pgd_alpha

        elif adv == 'rgb':
            perturb_sizes['rgb'] = [num_rays, 3]
            epsilons['rgb'] = pgd_eps
            alphas['rgb'] = pgd_alpha

    return perturb_sizes, epsilons, alphas, iters, norms

def train_one_epoch_adv(model, optimizer, scheduler, train_loader, test_set, exhibit_set,
    summary_writer, global_step, args, run_dir, device, awp_adversary):

    near, far = train_loader.dataset.near_far()

    start_step = global_step
    epoch = global_step // len(train_loader) + 1
    time0 = time.time()

    for (batch_rays, gt, cam_ids) in train_loader:

        # counter accumulate
        global_step += 1

        # make sure on cuda
        batch_rays, gt = batch_rays.to(device), gt.to(device)

        #####  Core optimization loop  #####

        # 1. Obtain adversarial sample
        model.eval()
        num_cameras = train_loader.dataset.num_images()
        H, W = train_loader.dataset.height_width()
        near, far = train_loader.dataset.near_far()
        perturb_sizes, epsilons, alphas, pgd_iters, pgd_norms = get_adv_templates(args.adv, H, W, near, far,
            num_cameras, batch_rays.shape[1], args)
        if args.adv_type == 'pgd':
            adv_perturb = attack_pgd(model, (batch_rays, (near, far), cam_ids), gt, perturb_sizes,
                epsilons, alphas, loss_fn=model.criterion, attack_iters=pgd_iters, norm=pgd_norms, unadv=args.unadv)
        elif args.adv_type == 'random':
            adv_perturb = attack_random(model, (batch_rays, (near, far), cam_ids), gt, perturb_sizes, epsilons, norm=pgd_norms)

        # 2. Apply weight perturbation if enabled
        if global_step >= args.awp_warmup and awp_adversary is not None:
            awp = awp_adversary.calc_awp(inputs=(batch_rays, (near, far)), targets=gt,
                delta=adv_perturb, loss_fn=model.criterion, unadv=args.unadv)
            awp_adversary.perturb(awp)

        # 3. Inject adversarial sample
        model.train()
        ret_adv = model(batch_rays, (near, far), cam_ids, adv_perturb=adv_perturb) # with adversarial
        ret_clean = model(batch_rays, (near, far), cam_ids) # without adversarial

        optimizer.zero_grad()

        loss_clean, loss_adv = model.criterion(ret_clean, gt), model.criterion(ret_adv, gt)
        loss = (1.0 - args.adv_lambda) * loss_clean + args.adv_lambda * loss_adv
        psnr = model.metric(ret_clean, gt)['psnr']

        # 4. Optimize model
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)

        # 5. Restore model within warm-up iterations
        if global_step >= args.awp_warmup and awp_adversary is not None:
            awp_adversary.restore(awp)

        ############################
        ##### Rest is logging ######
        ############################

        # logging errors
        if global_step % args.i_print == 0 and global_step > 0:
            dt = time.time() - time0
            time0 = time.time()
            avg_time = dt / min(global_step - start_step, args.i_print)
            print(f"[TRAIN] Iter: {global_step}/{args.max_steps} Loss: {loss.item()} PSNR: {psnr} Average Time: {avg_time}")

            # log training metric
            summary_writer.add_scalar('train/loss', loss, global_step)
            summary_writer.add_scalar('train/psnr', psnr, global_step)

            # log learning rate
            lr_groups = {}
            for i, param in enumerate(optimizer.param_groups):
                lr_groups['group_'+str(i)] = param['lr']
            summary_writer.add_scalars('l_rate', lr_groups, global_step)

        # logging images
        if global_step % args.i_img == 0 and global_step > 0:
            # Output test images to tensorboard
            ret_dict, metric_dict = eval_one_view(model, test_set[args.log_img_idx], (near, far), device=device)
            summary_writer.add_image('test/rgb', to8b(ret_dict['rgb'].numpy()), global_step, dataformats='HWC')
            summary_writer.add_image('test/disp', to8b(ret_dict['disp'].numpy() / np.max(ret_dict['disp'].numpy())), global_step, dataformats='HWC')

            # Render test set to tensorboard looply
            ret_dict, metric_dict = eval_one_view(model, test_set[(global_step//args.i_img-1) % len(test_set)], (near, far), device=device)
            summary_writer.add_image('loop/rgb', to8b(ret_dict['rgb'].numpy()), global_step, dataformats='HWC')
            summary_writer.add_image('loop/disp', to8b(ret_dict['disp'].numpy() / np.max(ret_dict['disp'].numpy())), global_step, dataformats='HWC')

        # save checkpoint
        if global_step % args.i_weights == 0 and global_step > 0:
            path = os.path.join(run_dir, 'checkpoints', '{:08d}.ckpt'.format(global_step))
            print('Checkpointing at', path)
            save_checkpoint(path, global_step, model, optimizer)

        # test images
        if global_step % args.i_testset == 0 and global_step > 0:
            print("Evaluating test images ...")
            save_dir = os.path.join(run_dir, 'testset_{:08d}'.format(global_step))
            os.makedirs(save_dir, exist_ok=True)
            metric_dict = evaluate(model, test_set, device=device, save_dir=save_dir)

            # log testing metric
            summary_writer.add_scalar('test/mse', metric_dict['mse'], global_step)
            summary_writer.add_scalar('test/psnr', metric_dict['psnr'], global_step)

        # exhibition video
        if global_step % args.i_video==0 and global_step > 0 and exhibit_set is not None:
            render_video(model, exhibit_set, device=device, save_dir=run_dir, suffix=str(global_step))

        # End training if finished
        if global_step >= args.max_steps:
            print(f'Train ends at global_step={global_step}')
            break

    return global_step
