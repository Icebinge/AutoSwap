import cv2
import os
from .data_utils import * 
from .base import BaseDataset
from lvis import LVIS

class LvisDataset(BaseDataset):
    def __init__(self, image_dir, json_path):
        self.image_dir = image_dir
        self.json_path = json_path
        lvis_api = LVIS(json_path)
        img_ids = sorted(lvis_api.imgs.keys())
        imgs = lvis_api.load_imgs(img_ids)
        anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]
        self.data = imgs
        self.annos = anns
        self.lvis_api = lvis_api
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 0

    def register_subset(self, path):
        data = os.listdir(path)
        data = [ os.path.join(path, i) for i in data if '.json' in i]
        self.data = self.data + data

    def get_sample(self, idx):
        # ==== get pairs =====
        image_name = self.data[idx]['coco_url'].split('/')[-1]
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)
        ref_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anno = self.annos[idx]
        obj_ids = []
        for i in range(len(anno)):
            obj = anno[i]
            area = obj['area']
            if area > 3600:
                obj_ids.append(i)
        assert len(anno) > 0
        obj_id = rng.choice(obj_ids)
        anno = anno[obj_id]
        ref_mask = self.lvis_api.ann_to_mask(anno)

        tar_image, tar_mask = ref_image.copy(), ref_mask.copy()
        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        return item_with_collage

    def __len__(self):
        return 20000



        
