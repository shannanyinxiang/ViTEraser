import os 
import os 
import cv2
import numpy as np
from tqdm import tqdm

def generate_mask_SCUTENS(args):
    gt_folder = os.path.join(args.data_root, 'all_gts')
    image_folder = os.path.join(args.data_root, 'image')
    mask_folder = os.path.join(args.data_root, 'mask')
    os.makedirs(mask_folder, exist_ok=True)

    gt_paths = [os.path.join(gt_folder, ele) for ele in os.listdir(gt_folder)]
    for gt_path in tqdm(gt_paths):
        image_name = os.path.basename(gt_path).replace('.txt', '.jpg')
        image_path = os.path.join(image_folder, image_name)
        image_shape = cv2.imread(image_path).shape

        mask = np.ones(image_shape, dtype=np.uint8) * 255
        polygons = open(gt_path, 'r').read().splitlines()
        polygons = [list(map(int, ele.split(','))) for ele in polygons]
        polygons = [np.array(ele).reshape(-1, 2) for ele in polygons]
        for polygon in polygons:
            cv2.fillPoly(mask, [polygon], color=(0, 0, 0))
        
        save_path = os.path.join(mask_folder, image_name)
        cv2.imwrite(save_path, mask)

if __name__ == '__main__':
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root')
    args = parser.parse_args()

    generate_mask_SCUTENS(args)
