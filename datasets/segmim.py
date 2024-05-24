import os
import cv2
import torch
import numpy as np

from PIL import Image
from pathlib import Path
from torchvision.transforms import Compose
from torch.utils.data import Dataset, ConcatDataset

from . import transforms as T

class SegMIMDataset(Dataset):
    def __init__(self, gt_root, img_root, transforms):
        self.gt_paths = [os.path.join(gt_root, _) for _ in os.listdir(gt_root)]
        self.gt_paths.sort()
        self.img_root = img_root

        self.img_exts = ['.jpg', '.gif', '.png']
        self.transforms = transforms 

    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        img_path = self._get_img_path(gt_path)

        img = Image.open(img_path).convert('RGB')
        text_pts = self._read_gt(gt_path)
        gt_mask = self._draw_text_mask(text_pts, (img.size[1], img.size[0], 3))

        data = {
            'image': img,
            'gt_mask': gt_mask,
            'image_path': img_path}
        
        if self.transforms:
            data = self.transforms(data)

        return data 
    
    def __len__(self):
        return len(self.gt_paths)
    
    def _get_img_path(self, gt_path):
        gt_name = os.path.splitext(os.path.basename(gt_path))[0]
        for img_ext in self.img_exts:
            img_path = os.path.join(self.img_root, gt_name + img_ext)
            if os.path.exists(img_path):
                return img_path
            img_path = os.path.join(self.img_root, gt_name.replace('gt_', '') + img_ext)
            if os.path.exists(img_path):
                return img_path
        raise ValueError(f'Cannot find img path corresponding to {gt_path}')
    
    def _read_gt(self, gt_path):
        lines = open(gt_path, 'r').read().splitlines()
        pts = [list(map(int, line.split(','))) for line in lines]
        return pts 
    
    def _draw_text_mask(self, pts, shape):
        mask = np.ones(shape, dtype=np.uint8) * 255 
        for pt in pts:
            cv2.fillPoly(mask, [np.array(pt).reshape(-1, 2)], color=(0, 0, 0))
        mask = Image.fromarray(mask).convert('1')
        return mask


def make_transform(image_set, args):
    transforms = []
    if image_set == 'train':
        transforms.append(T.RandomCrop(args.crop_min_ratio, args.crop_max_ratio, args.crop_prob))
        transforms.append(T.RandomHorizontalFlip(args.horizontal_flip_prob))
        transforms.append(T.RandomRotate(args.rotate_max_angle, args.rotate_prob))
    transforms.append(T.Resize((args.pix2pix_size, args.pix2pix_size)))
    transforms.append(T.ToTensor())
    return Compose(transforms)

class SegMIMTransform(object):
    def __init__(self, image_set, args):
        self.transforms = make_transform(image_set, args)
        self.mask_generator = T.MaskGenerator(
            input_size=args.pix2pix_size,
            mask_patch_size=args.random_mask_patch_size,
            model_patch_size=4,
            mask_ratio=args.random_mask_ratio
        )

    def __call__(self, data):
        data = self.transforms(data)
        mask = self.mask_generator()
        mask = torch.from_numpy(mask)

        data['mask'] = mask 
        return data

def build(image_set, args):
    root = Path(args.data_root)
    if image_set == 'train':
        dataset_names = args.train_dataset
    elif image_set == 'val':
        dataset_names = args.val_dataset
    
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name == 'art_train':
            img_root = root / 'ArT' / 'train_images'
            gt_root = root / 'ArT' / 'training_det_gts'
        elif dataset_name == 'ic13_train':
            img_root = root / 'ICDAR2013' / 'training_images'
            gt_root = root / 'ICDAR2013' / 'training_det_gts'
        elif dataset_name == 'ic15_train':
            img_root = root / 'ICDAR2015' / 'training_images'
            gt_root =  root / 'ICDAR2015' / 'training_det_gts'
        elif dataset_name == 'lsvt_train':
            img_root = root / 'LSVT' / 'training_images'
            gt_root = root / 'LSVT' / 'training_det_gts'
        elif dataset_name == 'mlt2017_train':
            img_root = root / 'MLT2017' / 'training_images'
            gt_root = root / 'MLT2017' / 'training_det_gts'
        elif dataset_name == 'rects_train':
            img_root = root / 'ReCTS' / 'training_images'
            gt_root = root / 'ReCTS' / 'training_det_gts'
        elif dataset_name == 'textocr_train':
            img_root = root / 'TextOCR' / 'trainval_images'
            gt_root = root / 'TextOCR' / 'training_det_gts'
        elif dataset_name == 'textocr_val':
            img_root = root / 'TextOCR' / 'trainval_images'
            gt_root = root / 'TextOCR' / 'val_det_gts'
        elif dataset_name == 'ic13_test': 
            img_root = root / 'ICDAR2013' / 'testing_images'
            gt_root = root / 'ICDAR2013' / 'testing_det_gts'
        else:
            raise ValueError
        
        if args.segmim_finetune:
            transforms = make_transform(image_set, args)
        else:
            transforms = SegMIMTransform(image_set, args)
        dataset = SegMIMDataset(gt_root, img_root, transforms)
        print(gt_root, len(dataset))
        datasets.append(dataset)
    
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return dataset