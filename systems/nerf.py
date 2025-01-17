import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.ray_utils import get_rays, get_ray_directions
from datasets.pose_optimization import multiply
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR


@systems.register('nerf-system')
class NeRFSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * self.config.model.num_samples_per_ray
        self.train_num_rays = self.config.model.train_num_rays
        self.automatic_optimization = False

    def forward(self, batch):
        return self.model(batch['rays'])
    
    def preprocess_data(self, batch, stage):
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,))
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,))
        if stage in ['train']:  
            x = torch.randint(0, self.dataset.w, size=(self.train_num_rays,))
            y = torch.randint(0, self.dataset.h, size=(self.train_num_rays,))
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(dtype=torch.float32) / 255
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(dtype=torch.float32)
            x, y, index = x.to(self.rank), y.to(self.rank), index.to(self.rank)
            c2w = self.dataset.all_c2w[index]
            
            #if dataset name is evdata, then we have different direction vectors for each image
            if self.dataset.config.name == 'evdata':
                c2w = multiply(c2w, self.dataset.pose_refine(index))
                K = self.dataset.all_K[index]
                directions = get_ray_directions(i=x, j=y, fx=K[:, 2], fy=K[:, 3], cx=K[:, 4], cy=K[:, 5])
                # directions = self.dataset.all_directions[index, y, x]
            else:
                directions = self.dataset.directions[y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            
        else:
            index = index.cpu()
            c2w = self.dataset.all_c2w[index][0]
            if self.dataset.config.name == 'evdata':
                #directions = self.dataset.all_directions[index][0]
                c2w = multiply(c2w, self.dataset.pose_refine(index.to(self.rank)).detach().cpu())
                K = self.dataset.all_K[index][0]
                directions = get_ray_directions(i=K[0], j=K[1], fx=K[2], fy=K[3], cx=K[4], cy=K[5], test=True)
            else:
                directions = self.dataset.directions

            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(dtype=torch.float32) / 255
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(dtype=torch.float32)
        
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        rays, rgb, fg_mask = rays.to(self.rank), rgb.to(self.rank), fg_mask.to(self.rank)
        rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])
        
        batch.update({
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask
        })
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)
        
        loss_rgb = F.smooth_l1_loss(out['comp_rgb'][out['rays_valid']], batch['rgb'][out['rays_valid']])
        self.log('train/loss_rgb', loss_rgb)
        loss += loss_rgb * self.C(self.config.system.loss.lambda_rgb)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss, but still slows down training by ~30%
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        # Backward call and step optimizers
        if len(self.config.system.optimizer) > 1:
            [opt.zero_grad() for opt in self.optimizers()]
            self.manual_backward(loss)
            [opt.step() for opt in self.optimizers()]
        else:
            self.optimizers().zero_grad()
            self.manual_backward(loss)
            self.optimizers().step()

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))
        
        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)
        self.log('loss', loss.detach(), prog_bar=True)
        psnr = -10 * torch.log10(torch.mean(loss.detach()))
        self.log('psnr', psnr, prog_bar=True)

        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb'], batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'grayscale', 'img': out['opacity'].view(H, W), 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)         

    def test_step(self, batch, batch_idx):  
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb'], batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'grayscale', 'img': out['opacity'].view(H, W), 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }      
    
    def test_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    

            # self.save_img_sequence(
            #     f"it{self.global_step}-test",
            #     f"it{self.global_step}-test",
            #     '(\d+)\.png',
            #     save_format='mp4',
            #     fps=30
            # )
            
            mesh = self.model.isosurface()

            mesh['v_pos'] = mesh['v_pos'] * self.dataset.scene_scale_factor
            self.save_mesh(
                f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.ply",
                mesh['v_pos'],
                mesh['t_pos_idx'],
            )
