import cv2
import numpy as np
import os
from PIL import Image
from .data_utils import * 
from .base import BaseDataset

class DresscodeDataset(BaseDataset):
    def __init__(self, image_dir):
        self.image_root =  image_dir 
        self.data = os.listdir(self.image_root)
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 2

    def __len__(self):
        return 20000

    def get_sample(self, idx):
        tar_mask_path = os.path.join(self.image_root, self.data[idx])
        tar_image_path = tar_mask_path.replace('label_maps/','images/').replace('_4.png','_0.jpg')
        ref_image_path = tar_mask_path.replace('label_maps/','images/').replace('_4.png','_1.jpg')

        # Read Image and Mask
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        tar_image = cv2.imread(tar_image_path)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)

        ref_mask = (ref_image < 240).astype(np.uint8)[:,:,0]
        

        tar_mask = Image.open(tar_mask_path ).convert('P')
        tar_mask= np.array(tar_mask)
        tar_mask = tar_mask == 4


        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask, max_ratio = 1.0)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        return item_with_collage

