import cv2
import numpy as np
from torch.utils.data import Dataset
from .data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A

class BaseDataset(Dataset):
    def __init__(self):
        self.data = []

    def __len__(self):
        # We adjust the ratio of different dataset by setting the length.
        pass

    def aug_data_back(self, image):
        transform = A.Compose([
            A.HorizontalFlip(p = 0.5), 
            A.RandomBrightnessContrast(p = 0.5), 
            #A.Rotate(limit = 20, border_mode = cv2.BORDER_CONSTANT, value = (0, 0, 0)),
        ])

        transformed = transform(image = image.astype(np.uint8), mask = mask)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]

        return transformed_image, transformed_mask
    
    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        H, W = image.shape[0], image.shape[1]
        H, W = H * ratio, W * ratio
        y1, y2, x1, x2 = yyxx
        h, w = y2 - y1, x2 - x1
        if mode == 'max':
            return h <= H and w <= W
        elif mode == 'min':
            return h >= H and w >= W
        return True
    
    def __getitem__(self, idx):
        while True:
            try:
                index = rng.integers(0, len(self.data) - 1)
                item = self.get_sample(index)
                return item
            except Exception:
                index = rng.integers(0, len(self.data) - 1)
    
    def get_sample(self, index):
        # Implemented for each specific dataset
        pass

    def sample_timestep(self, max_step = 1000):
        if rng.random() < 0.3:
            step = rng.integers(0, max_step)
            return np.array([step])
        
        step_start = 0
        step_end = max_step

        if self.dynamic == 1:
            # coarse videos
            step_start = max_step // 2
        elif self.dynamic == 0:
            # static images
            step_end = max_step // 2
        
        step = rng.integers(step_start, step_end)
        return np.array([step])

    def check_mask_area(self, mask):
        H, W = mask.shape[0], mask.shape[1]
        ratio = mask.sum() / (H * W)
        return 0.1 * 0.1 <= ratio <= 0.8 * 0.8
    
    def process_pairs(self, ref_image, ref_mask, tar_image, tar_mask, max_ratio = 0.8):
        assert mask_score(ref_mask) > 0.90
        assert self.check_mask_area(ref_mask)
        assert self.check_mask_area(tar_mask)

        # Get the outline Box of the reference image
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        assert self.check_region_size(ref_mask, ref_box_yyxx, ratio = 0.10, mode = 'min')

        # Filtering background for the reference image
        ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1 - ref_mask_3)

        y1, y2, x1, x2 = ref_box_yyxx
        masked_ref_image = masked_ref_image[y1 : y2, x1 : x2, : ]
        ref_mask = ref_mask[y1 : y2, x1 : x2]

        ratio = rng.integers(11, 15) / 10
        masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio = ratio)
        ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)

        # Padding reference image to square and resize to 224
        masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
        masked_ref_image  = cv2.resize(masked_ref_image.astype(np.uint8), (224, 224)).astype(np.uint8)
        ref_mask = ref_mask_3[:, :, 0]

        # Augmenting reference image
        # masked_ref_image_aug = self.aug_data(masked_ref_image)

        # Getting for high-freqency map
        masked_ref_image_compose, ref_mask_compose = self.aug_data_mask(masked_ref_image, ref_mask)
        masked_ref_image_aug = masked_ref_image_compose.copy()

        ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose / 255)

        # ============ Training Target ============
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio = [1.1, 1.2])
        assert self.check_region_size(tar_mask, tar_box_yyxx, ratio = max_ratio, mode = 'max')

        # Cropping around the target object
        tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio = [1.3, 3.0])
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) 
        y1, y2, x1, x2 = tar_box_yyxx_crop
        cropped_target_image = tar_image[y1:y2, x1:x2, :]
        cropped_tar_mask = tar_mask[y1:y2, x1:x2]
        tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
        y1, y2, x1, x2 = tar_box_yyxx

        # Prepairing collage image
        ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2 - x1, y2 - y1))

        collage = cropped_target_image.copy()
        collage[y1:y2, x1:x2] = ref_image_collage

        collage_mask = cropped_target_image.copy()
        collage_mask[y1:y2, x1:x2] = 1.0

        if rng.uniform(0, 1) < 0.7:
            cropped_tar_mask = perturb_mask(cropped_tar_mask)
            collage_mask = np.stack([cropped_tar_mask] * 3, -1)

        H1, W1 = collage.shape[0], collage.shape[1]

        cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0).astype(np.uint8)
        collage = pad_to_square(collage, pad_value = 0).astype(np.uint8)
        collage_mask = pad_to_square(collage_mask, pad_value = 2).astype(np.uint8)
        H2, W2 = collage.shape[0], collage.shape[1]

        cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512, 512)).astype(np.float32)
        collage = cv2.resize(collage.astype(np.uint8), (512, 512)).astype(np.float32)
        collage_mask = cv2.resize(collage_mask.astype(np.uint8), (512, 512), interpolation = cv2.INTER_NEAREST).astype(np.float32)
        collage_mask[collage_mask == 2] = -1

        # Prepairing dataloader items
        masked_ref_image_aug /= 255
        cropped_target_image = cropped_target_image / 127.5 - 1.0
        collage = collage / 127.5 - 1.0
        collage = np.concatenate([collage, collage_mask[:, :, :1]], -1)

        item = dict(
            ref = masked_ref_image_aug.copy(), 
            jpg = cropped_target_image.copy(),
            hint = collage.copy(), 
            extra_size = np.array([H1, W1, H2, W2]),
            tar_box_yyxx_crop = np.array(tar_box_yyxx_crop)
        )
    
        return item
