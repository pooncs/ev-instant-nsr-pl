import os
import json
import math
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank


class EVDatasetBase():
    def setup(self, config, split):
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

        self.all_c2w, self.all_images, self.all_fg_masks, self.all_directions = [], [], [], []

        for i, frame in enumerate(meta['frames']):
            c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
            self.all_c2w.append(c2w)

            img_path = os.path.join(self.config.root_dir, frame['file_path'][2:])
            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)

            direction = get_ray_directions(self.w, self.h, frame['fl_x'], frame['fl_y'], frame['cx'], frame['cy'], self.config.use_pixel_centers).to(self.rank) # (h, w, 3)
            self.all_directions.append(direction)

            self.all_fg_masks.append(img[..., -1]) # (h, w)
            self.all_images.append(img[...,:3])

        self.all_c2w, self.all_images, self.all_fg_masks, self.all_directions = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float().to(self.rank), \
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank), \
            torch.stack(self.all_directions, dim=0).float().to(self.rank)
        

class EVDataset(Dataset, EVDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class EVIterableDataset(IterableDataset, EVDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('evdata')
class EVDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = EVIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = EVDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = EVDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = EVDataset(self.config, self.config.train_split)

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
