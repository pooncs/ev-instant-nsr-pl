import os
import json
import math
import numpy as np
from PIL import Image
import imageio.v3 as imageio
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank
from datasets.utils import getROICornerPixels, maskImage
from datasets.pose_optimization import CameraOptimizer


class EVDatasetBase():
    def setup(self, config, split, pose_refine: CameraOptimizer=None):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.use_mask = True

        with open(os.path.join(self.config.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)

        w = meta['frames'][0]['w']
        h = meta['frames'][0]['h']

        # if 'w' in meta and 'h' in meta:
        #     W, H = int(meta['w']), int(meta['h'])
        # else:
        #     W, H = 800, 800

        # if 'img_wh' in self.config:
        #     w, h = self.config.img_wh
        #     assert round(W / w * h) == H
        # elif 'img_downscale' in self.config:
        #     w, h = W // self.config.img_downscale, H // self.config.img_downscale
        # else:
        #     raise KeyError("Either img_wh or img_downscale should be specified.")
        
        self.w, self.h = w, h
        self.img_wh = (self.w, self.h)

        self.near, self.far = self.config.near_plane, self.config.far_plane

        # self.focal = 0.5 * w / math.tan(0.5 * meta['camera_angle_x']) # scaled focal length

        # ray directions for all pixels, same for all images (same H, W, focal)
        # self.directions = get_ray_directions(self.w, self.h, self.focal, self.focal, self.w//2, self.h//2, self.config.use_pixel_centers).to(self.rank) # (h, w, 3)           
        aabb = meta['aabb']
        Pw = np.array([[aabb[0][0],aabb[1][0],aabb[0][0],aabb[1][0]],[aabb[1][1],aabb[1][1],aabb[0][1],aabb[0][1]],[0,0,0,0],[1,1,1,1]])
        
        self.scene_scale_factor = meta["aabb"][1][0] #we use xmax in aabb to downscale scene to unit cube [-1,1] 

        self.all_c2w, self.all_images, self.all_fg_masks, self.all_K = [], [], [], []
        
        for frame in tqdm(meta['frames']):
            c2w = torch.from_numpy(np.array(frame['transform_matrix']))
            pts = getROICornerPixels(c2w, frame['fl_x'], frame['cx'], frame['cy'], w, Pw)
            c2w[0:3,3] /= self.scene_scale_factor # scale to unit cube
            self.all_c2w.append(c2w[:3, :4])

            img_path = os.path.join(self.config.root_dir, frame['file_path'])
            # img = Image.open(img_path)
            # img = img.resize(self.img_wh, Image.BICUBIC)
            # img, _ = maskImage(np.array(img), pts)
            # img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
            # img = (img * 255).to(dtype=torch.uint8)

            img = imageio.imread(img_path, extension='.png')
            img, _ = maskImage(img, pts)
            img = torch.tensor(img, dtype=torch.uint8)

            #direction = get_ray_directions(self.w, self.h, frame['fl_x'], frame['fl_y'], frame['cx'], frame['cy'], self.config.use_pixel_centers) # (h, w, 3)
            #self.all_directions.append(direction)
            self.all_K.append(torch.tensor([frame['w'], frame['h'], frame['fl_x'], frame['fl_y'], frame['cx'], frame['cy']]))
            
            self.all_fg_masks.append(img[..., -1].to(dtype=torch.bool)) # (h, w)
            self.all_images.append(img[...,:3])

        self.all_images = torch.stack(self.all_images, dim=0)
        self.all_fg_masks = torch.stack(self.all_fg_masks, dim=0)
        
        self.all_c2w = torch.stack(self.all_c2w, dim=0).float()
        self.all_K = torch.stack(self.all_K, dim=0).float()

        if split == 'train':
            self.all_c2w = self.all_c2w.to(device=self.rank)
            self.all_K = self.all_K.to(device=self.rank)
            self.pose_refine = CameraOptimizer(self.config.pose_refine, self.all_c2w.shape[0], self.rank)
        else:
            self.pose_refine = pose_refine

        #self.all_directions = torch.stack(self.all_directions, dim=0).float().to(device=self.rank)
        

class EVDataset(Dataset, EVDatasetBase):
    def __init__(self, config, split, pose_refine):
        self.setup(config, split, pose_refine)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class EVIterableDataset(IterableDataset, EVDatasetBase):
    def __init__(self, config, split, pose_refine):
        self.setup(config, split, pose_refine)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('evdata')
class EVDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        with open(os.path.join(config.root_dir, f"transforms_train.json"), 'r') as f:
            self.num_cameras = len(json.load(f)['frames'])
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = EVIterableDataset(self.config, self.config.train_split, None)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = EVDataset(self.config, self.config.val_split, self.train_dataset.pose_refine)
        if stage in [None, 'test']:
            self.test_dataset = EVDataset(self.config, self.config.test_split, self.train_dataset.pose_refine)
        if stage in [None, 'predict']:
            self.predict_dataset = EVDataset(self.config, self.config.train_split, self.train_dataset.pose_refine)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
