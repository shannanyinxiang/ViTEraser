import os 
import cv2
import torch 

from tqdm import tqdm 
from utils.visualize import tensor_to_cv2image

@torch.no_grad()
def evaluate(model, dataloader, args):
    model.eval()

    device = torch.device(args.device)

    for data in tqdm(dataloader):
        images = data['image'].to(device)

        outputs = model(images)[-1]

        for i, output in enumerate(outputs):
            image_path = data['image_path'][i]
            dataset_name = get_dataset_name(image_path)
            save_folder = os.path.join(args.output_dir, dataset_name)
            os.makedirs(save_folder, exist_ok=True)

            image_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(save_folder, image_name + '.png')
            output = torch.clamp(output, min=0, max=1)
            output = tensor_to_cv2image(output.cpu(), False)
            cv2.imwrite(save_path, output)
                

def get_dataset_name(image_path):
    if 'scut-syn' in image_path.lower():
        dataset_name = 'SCUT-Syn'
    elif 'scut-ens' in image_path.lower():
        dataset_name = 'SCUT-EnsText'
    elif 'casia-str' in image_path.lower():
        dataset_name = 'CASIA-STR'
    else:
        raise ValueError(f'Unknown dataset for {image_path}')
    return dataset_name 