import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from models.camera import transform_rays
from models.sampler import StratifiedSampler, ImportanceSampler
from models.renderer import VolumetricRenderer
from models.nerf_mlp import NeRFMLP
from models.nerf_net import NeRFNet

from utils.error import *
from utils.image import img2mse, mse2psnr

class AdvNeRFNet(NeRFNet):
    
    def __init__(self, netdepth=8, netwidth=256, netdepth_fine=8, netwidth_fine=256, N_samples=64, N_importance=64,
        viewdirs=True, use_embed=True, multires=10, multires_views=4, conv_embed=False, ray_chunk=1024*32, pts_chuck=1024*64,
        perturb=1., raw_noise_std=0., white_bkgd=False):
        
        super().__init__(netdepth, netwidth, netdepth_fine, netwidth_fine, N_samples, N_importance,
            viewdirs, use_embed, multires, multires_views, conv_embed, ray_chunk, pts_chuck,
            perturb=0., raw_noise_std=0.) # IMPORTANT: no random perturbation

        self.nerf_adv = NeRFMLP(input_dim=3, output_dim=4, net_depth=netdepth_fine, net_width=netwidth_fine, skips=[4],
            viewdirs=viewdirs, use_embed=use_embed, multires=multires, multires_views=multires_views,
            conv_embed=conv_embed, netchunk=pts_chuck)

    def render_rays(self, rays_o, rays_d, near, far, viewdirs=None, raw_noise_std=0.,
        verbose=False, retraw = False, retpts=False, adv_perturb={}, **kwargs):
        """Volumetric rendering.
        Args:
          ray_o: origins of rays. [N_rays, 3]
          ray_d: directions of rays. [N_rays, 3]
          near: the minimal distance. [N_rays, 1]
          far: the maximal distance. [N_rays, 1]
          raw_noise_std: If True, add noise on raw output from nn
          verbose: bool. If True, print more debugging info.
        Returns:
          rgb: [N_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
          raw: [N_rays, N_samples, C]. Raw predictions from model.
          pts: [N_rays, N_samples, 3]. Sampled points.
          rgb0: See rgb_map. Output for coarse model.
          raw0: See raw. Output for coarse model.
          pts0: See acc_map. Output for coarse model.
          z_std: [N_rays]. Standard deviation of distances along ray for each sample.
        """
        bounds = torch.cat([near, far], -1) # [N_rays, 2]

        # Primary sampling
        perturb_pts, perturb_zs = adv_perturb.get('pts_c', None), adv_perturb.get('zval_c', None)
        pts, z_vals, _ = self.point_sampler(rays_o, rays_d, bounds,
            adv_pts=perturb_pts, adv_zs=perturb_zs, **kwargs)  # [N_rays, N_samples, 3]
        viewdirs_c = viewdirs[..., None, :].expand(pts.shape) # [N_rays, 3] -> [N_rays, N_samples, 3]

        # obtain raw data
        perturb_feat = adv_perturb.get('feat_c', None)
        if len(adv_perturb) > 0:
            raw = self.nerf_adv(pts, viewdirs_c, adv_feat=perturb_feat)
        else:
            raw = self.nerf(pts, viewdirs_c)

        # render raw data
        perturb_raw = adv_perturb.get('raw_c', None)
        ret = self.renderer(raw, z_vals, rays_d, adv_raw=perturb_raw, raw_noise_std=raw_noise_std)

        # Buffer raw/pts
        if retraw:
            ret['raw'] = raw
        if retpts:
            ret['pts'] = pts
        
        # Secondary sampling
        N_importance = kwargs.get('N_importance', self.N_importance)
        if (self.importance_sampler is not None) and (N_importance > 0):
            # backup coarse model output
            ret0 = ret

            # resample
            perturb_pts, perturb_zs = adv_perturb.get('pts_f', None), adv_perturb.get('zval_f', None)
            pts, z_vals, sampler_extras = self.importance_sampler(rays_o, rays_d, z_vals, **ret,
                adv_pts=perturb_pts, adv_zs=perturb_zs, **kwargs) # [N_rays, N_samples + N_importance, 3]
            viewdirs_f = viewdirs[..., None, :].expand(pts.shape) # [N_rays, 3] -> [N_rays, N_samples, 3]

            # obtain raw data
            perturb_feat = adv_perturb.get('feat_f', None)
            if len(adv_perturb) > 0:
                raw = self.nerf_adv(pts, viewdirs_f, adv_feat=perturb_feat)
            else:
                raw = self.nerf_fine(pts, viewdirs_f)

            # render raw data
            perturb_raw = adv_perturb.get('raw_f', None)
            ret = self.renderer(raw, z_vals, rays_d, adv_raw=perturb_raw, raw_noise_std=raw_noise_std)
            
            # Buffer raw/pts
            if retraw:
                ret['raw'] = raw
            if retpts:
                ret['pts'] = pts

            # compute std of resampled point along rays
            ret['z_std'] = torch.std(sampler_extras['z_samples'], dim=-1, unbiased=False)  # [N_rays]

            # buffer coarse model output
            for k in ret0:
                ret[k+'0'] = ret0[k]

        return ret

    def forward(self, ray_batch, bound_batch, cam_ids=None, adv_perturb={}, **kwargs):
        """Render rays
        Args:
          ray_batch: array of shape [2, batch_size, 3]. Ray origin and direction for
            each example in batch.
        Returns:
          ret_all includes the following returned values:
          rgb_map: [batch_size, 3]. Predicted RGB values for rays.
          raw: [batch_size, N_sample, C]. Raw data of each point.
          weight_map: [batch_size, N_sample, C]. Convert raw to weight scale (0-1).
          acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
        """

        # Render settings
        if self.training:
            render_kwargs = self.render_kwargs_train.copy()
            render_kwargs.update(kwargs)
        else:
            render_kwargs = self.render_kwargs_test.copy()
            render_kwargs.update(kwargs)

        # Disentangle ray batch
        rays_o, rays_d = ray_batch
        assert rays_o.shape == rays_d.shape

        # Flatten ray batch
        old_shape = rays_d.shape # [..., 3(+id)]
        rays_o = torch.reshape(rays_o, [-1,rays_o.shape[-1]]).float()
        rays_d = torch.reshape(rays_d, [-1,rays_d.shape[-1]]).float()

        # Adv perturb on cam pose
        adv_cam_r, adv_cam_t = adv_perturb.get('cam_r', None), adv_perturb.get('cam_t', None)
        if adv_cam_r is not None and adv_cam_t is not None:
            rays_o, rays_d = transform_rays(rays_o, rays_d, cam_ids, adv_cam_r, adv_cam_t)

        # Compute ray directions as input
        if self.use_viewdirs: 
            viewdirs = rays_d
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, viewdirs.shape[-1]]).float()  

        # Disentangle bound batch
        near, far = bound_batch
        if isinstance(near, int) or isinstance(near, float):
            near = near * torch.ones_like(rays_d[..., :1], dtype=torch.float)
        if isinstance(far, int) or isinstance(far, float):
            far = far * torch.ones_like(rays_d[..., :1], dtype=torch.float)

        # Batchify rays
        all_ret = {}
        for i in range(0, rays_o.shape[0], self.chunk):
            end = min(i+self.chunk, rays_o.shape[0])
            chunk_o, chunk_d = rays_o[i:end], rays_d[i:end]
            chunk_n, chunk_f = near[i:end], far[i:end]
            chunk_v = viewdirs[i:end] if self.use_viewdirs else None

            chunk_p = {}
            for k, v in adv_perturb.items():
                if v.shape[0] == rays_o.shape[0]:
                    chunk_p[k] = v[i:end]
                else:
                    chunk_p[k] = v

            # Render function
            ret = self.render_rays(chunk_o, chunk_d, chunk_n, chunk_f, viewdirs=chunk_v,
                adv_perturb=chunk_p, **render_kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}

        # Adv perturb on output
        adv_rgb = adv_perturb.get('rgb', None)
        if adv_rgb is not None:
            all_ret['rgb'] += adv_rgb

        # Unflatten
        for k in all_ret:
            k_sh = list(old_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh) # [input_rays_shape, per_ray_output_shape]

        return all_ret

    def criterion(self, ret_dict, y, reduction='mean'):
        rgb = ret_dict['rgb']
        diff = torch.mean((rgb - y) ** 2, -1)
        img_loss = img2mse(rgb, y, reduction=reduction)

        loss = img_loss
        if 'rgb0' in ret_dict:
            img_loss0 = img2mse(ret_dict['rgb0'], y, reduction=reduction)
            loss = loss + img_loss0

        return loss

    def metric(self, ret_dict, y):
        rgb = ret_dict['rgb']
        mse = img2mse(rgb, y)
        psnr = mse2psnr(mse)

        metric_dict = {'mse': mse.item(), 'psnr': psnr.item()}

        if 'rgb0' in ret_dict:
            mse0 = img2mse(ret_dict['rgb0'], y)
            psnr0 = mse2psnr(mse0)
            metric_dict['mse0'] = mse0.item()
            metric_dict['psnr0'] = psnr0.item()

        return metric_dict
     

