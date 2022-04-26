import os, sys
import numpy as np
import imageio
import json
import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from models.embedder import Embedder

from utils.error import *

class MLP(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """
        MLP backbone for NeRF
        """
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        if use_viewdirs:
            self.alpha_linear = nn.Linear(W, 1)
            self.feature_linear = nn.Linear(W, W)
            ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

            ### Implementation according to the paper
            # self.views_linears = nn.ModuleList(
            #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

            self.rgb_linear = nn.Linear(W//2, output_ch-1)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, adv_feat=None):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if adv_feat is not None:
            h = h + adv_feat

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)

            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))

# Point query with embedding
class NeRFMLP(nn.Module):
    
    def __init__(self, input_dim=3, output_dim=4, net_depth=8, net_width=256, skips=[4],
        viewdirs=True, use_embed=True, multires=10, multires_views=4, conv_embed=False, netchunk=1024*64):

        super().__init__()

        self.chunk = netchunk

        # Provide empty periodic_fns to specify identity embedder
        periodic_fns = []
        if use_embed:
            periodic_fns = [torch.sin, torch.cos]

        self.embedder = Embedder(input_dim, multires, multires-1, periodic_fns, log_sampling=True, include_input=True)
        input_ch = self.embedder.out_dim

        input_ch_views = 0
        self.embeddirs = None
        if viewdirs:
            self.embeddirs = Embedder(input_dim, multires_views, multires_views-1, periodic_fns, log_sampling=True, include_input=True)
            input_ch_views = self.embeddirs.out_dim

        kernel_size = 3
        padding = math.floor(kernel_size/2)
        self.conv_embed, self.conv_embeddirs = None, None
        if conv_embed:
            self.conv_embed = nn.Conv1d(input_ch, input_ch, kernel_size=kernel_size, padding=padding)
            if self.embeddirs is not None:
                self.conv_embeddirs = nn.Conv1d(input_ch_views, input_ch_views, kernel_size=kernel_size, padding=padding)

        output_ch = output_dim
        self.mlp = MLP(net_depth, net_width, skips=skips, input_ch=input_ch,
            output_ch=output_ch, input_ch_views=input_ch_views, use_viewdirs=viewdirs)

    def batchify(self, inputs):
        """Single forward feed that applies to smaller batches.
        """
        query_batches = []
        for i in range(0, inputs.shape[0], self.chunk):
            end = min(i+self.chunk, inputs.shape[0])
            h = self.mlp(inputs[i:end]) # [N_chunk, C]
            query_batches.append(h)
        outputs = torch.cat(query_batches, 0) # [N_pts, C]
        return outputs

    def forward(self, inputs, viewdirs=None, adv_feat=None, **kwargs):
        """Prepares inputs and applies network.
        """
        # Flatten
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]) # [N_pts, C]
        if viewdirs is not None:
            # input_dirs = viewdirs[:,None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(viewdirs, [-1, viewdirs.shape[-1]])
        if adv_feat is not None:
            adv_feat = torch.reshape(adv_feat, [-1, adv_feat.shape[-1]])

        # Batchify
        output_chunks = []
        for i in range(0, inputs_flat.shape[0], self.chunk):
            end = min(i+self.chunk, inputs_flat.shape[0])

            embedded = self.embedder(inputs_flat[i:end])
            # apply 1d conv
            if self.conv_embed is not None:
                N_dim = embedded.shape[-1]
                embedded = embedded.view([-1, inputs.shape[1], N_dim]) # [N_chunk * N_sample, dim] -> [N_chunk, N_sample, dim]
                embedded = self.conv_embed(embedded.permute([0, 2, 1])) # [N_chunk, dim, N_sample]
                embedded = embedded.permute([0, 2, 1]).reshape([-1, N_dim]) # [N_chunk, N_sample, dim] -> [N_chunk * N_sample, dim]
            # append view direction embedding
            if self.embeddirs is not None:
                embedded_dirs = self.embeddirs(input_dirs_flat[i:end])
                if self.conv_embeddirs is not None:
                    N_dim = embedded_dirs.shape[-1]
                    embedded_dirs = embedded_dirs.view([-1, inputs.shape[1], N_dim]) # [N_chunk * N_sample, dim] -> [N_chunk, N_sample, dim]
                    embedded_dirs = self.conv_embeddirs(embedded_dirs.permute([0, 2, 1])) # [N_chunk, dim, N_sample]
                    embedded_dirs = embedded_dirs.permute([0, 2, 1]).reshape([-1, N_dim]) # [N_chunk, N_sample, dim] -> [N_chunk * N_sample, dim]
                embedded = torch.cat([embedded, embedded_dirs], -1)

            if adv_feat is not None:
                h = self.mlp(embedded, adv_feat[i:end]) # [N_chunk, C]
            else:
                h = self.mlp(embedded) # [N_chunk, C]
            output_chunks.append(h)
        outputs_flat = torch.cat(output_chunks, 0) # [N_pts, C]

        # Unflatten
        sh = list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
        return torch.reshape(outputs_flat, sh)
