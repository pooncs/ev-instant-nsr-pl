import torch
import numpy as np


def cast_rays(ori, dir, z_vals):
    return ori[..., None, :] + z_vals[..., None] * dir[..., None, :]


def get_ray_directions(i, j, fx, fy, cx, cy, use_pixel_centers=True, test=False):
    pixel_center = 0.5 if use_pixel_centers else 0
    
    if test:
        i, j = np.meshgrid(
            np.arange(i, dtype=np.float32),
            np.arange(j, dtype=np.float32),
            indexing='xy'
        )
        i, j = torch.from_numpy(i), torch.from_numpy(j)

    directions = torch.stack([(i - cx + pixel_center) / fx, 
                              -(j - cy + pixel_center) / fy, 
                              -torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def get_rays(directions, c2w, keepdim=False):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    assert directions.shape[-1] == 3

    if directions.ndim == 2: # (N_rays, 3)
        assert c2w.ndim == 3 # (N_rays, 4, 4) / (1, 4, 4)
        rays_d = (directions[:,None,:] * c2w[:,:3,:3]).sum(-1) # (N_rays, 3)
        rays_o = c2w[:,:,3].expand(rays_d.shape)
    elif directions.ndim == 3: # (H, W, 3)
        if c2w.ndim == 2: # (4, 4)
            rays_d = (directions[:,:,None,:] * c2w[None,None,:3,:3]).sum(-1) # (H, W, 3)
            rays_o = c2w[None,None,:,3].expand(rays_d.shape)
        elif c2w.ndim == 3: # (B, 4, 4)
            rays_d = (directions[None,:,:,None,:] * c2w[:,None,None,:3,:3]).sum(-1) # (B, H, W, 3)
            rays_o = c2w[:,None,None,:,3].expand(rays_d.shape)

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d
