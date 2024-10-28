import os
import math
import random
import importlib
from PIL import Image
from glob import glob
from tqdm import tqdm
from itertools import islice
from pytorch_msssim import ssim

from utils.option import args

import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torchvision.utils import save_image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, size, shuffle=False):
        super(Dataset, self).__init__()
        img_tf = transforms.Compose(
            [#transforms.CenterCrop(size=(178, 178)), # use this for CelebA only
            transforms.Resize(size=(size, size),
            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        mask_tf = transforms.Compose(
            [transforms.Resize(size=size, interpolation=transforms.InterpolationMode.NEAREST),
             transforms.ToTensor()])

        self.img_transform = img_tf
        self.mask_transform = mask_tf
        self.shuffle = shuffle

        self.paths = sorted(glob('{:s}/*'.format(img_root)))
        self.mask_paths = sorted(glob('{:s}/*.png'.format(mask_root)))
        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        if self.shuffle:
            mask_idx = random.randint(0, self.N_mask - 1)
        else:
            mask_idx = index if index < self.N_mask else index % (self.N_mask - 1)
        mask = Image.open(self.mask_paths[mask_idx])
        mask = self.mask_transform(mask.convert('L'))

        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        return gt_img, mask

    def __len__(self):
        return len(self.paths)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def L1(x, y):
    return nn.L1Loss()(x, y).item()

def normalize(x):
    x = x.transpose(1, 3) # [-1, 1]
    mean = torch.Tensor([1/2, 1/2, 1/2]).to(x.device)
    std = torch.Tensor([1/2, 1/2, 1/2]).to(x.device)
    x = x * std + mean # [0, 1]
    std = torch.Tensor([255, 255, 255]).to(x.device)
    x = x * std # [0, 255]
    x = x.transpose(1, 3)
    return x

def main_worker(args):
    # Model and version
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator(args).cuda()
    model.load_state_dict(torch.load(args.pre_train, map_location='cuda'))
    model.eval()

    # Prepare dataset
    dataset = Dataset(args.dir_image, args.dir_mask, args.image_size)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=args.shuffle, num_workers=4)
    image_data_loader = sample_data(dataloader)

    os.makedirs(args.outputs, exist_ok=True)
    os.makedirs(os.path.join(args.outputs, 'comp_results'), exist_ok=True)
    os.makedirs(os.path.join(args.outputs, 'gts'), exist_ok=True)

    # Initialize metrics for each mask ratio category
    metrics = {
        '0-20%': {'L1': 0, 'PSNR': 0, 'SSIM': 0, 'count': 0},
        '20-40%': {'L1': 0, 'PSNR': 0, 'SSIM': 0, 'count': 0},
        '40-60%': {'L1': 0, 'PSNR': 0, 'SSIM': 0, 'count': 0}
    }

    # Iteration through datasets
    for idx in tqdm(range(args.num_test)):
        image, mask = next(image_data_loader)
        image, mask = image.cuda(), mask.cuda()
        image_masked = image * (1 - mask).float() + mask

        with torch.no_grad():
            pred_img = model(image_masked, mask)

        comp_imgs = (1 - mask) * image + mask * pred_img

        # Calculate mask ratio
        mask_ratio = mask.sum() / mask.numel()  # Total masked pixels / total pixels

        # Determine the category based on mask ratio
        if mask_ratio <= 0.2:
            category = '0-20%'
        elif mask_ratio <= 0.4:
            category = '20-40%'
        elif mask_ratio <= 0.6:
            category = '40-60%'
        else:
            continue  # Skip if the mask ratio is above 60%

        # Update metrics for the appropriate category
        L1error = L1(comp_imgs, image)
        metrics[category]['L1'] += L1error

        # Calculate PSNR and SSIM
        comp_imgs_normalized = normalize(comp_imgs)
        image_normalized = normalize(image)
        ssim_result = ssim(comp_imgs_normalized, image_normalized, data_range=255, size_average=True).item()
        mse = torch.pow(comp_imgs_normalized - image_normalized, 2).mean().item()
        psnr_result = 10 * math.log10(255**2 / mse)

        # Store the metrics in the appropriate category
        metrics[category]['PSNR'] += psnr_result
        metrics[category]['SSIM'] += ssim_result
        metrics[category]['count'] += 1

        # Save images as before
        save_image(torch.cat([image_masked, comp_imgs, image], 0), os.path.join(args.outputs, f'{idx}_all.jpg'), nrow=3, normalize=True, value_range=(-1, 1))
        save_image(mask, os.path.join(args.outputs, f'{idx}_hole.jpg'), normalize=True, value_range=(0, 1))
        save_image(image_masked, os.path.join(args.outputs, f'{idx}_masked.jpg'), normalize=True, value_range=(-1, 1))
        save_image(comp_imgs, os.path.join(args.outputs, f'{idx}_comp.jpg'), normalize=True, value_range=(-1, 1))
        save_image(comp_imgs, os.path.join(args.outputs, 'comp_results', f'{idx}.png'), normalize=True, value_range=(-1, 1))
        save_image(image, os.path.join(args.outputs, f'{idx}_gt.jpg'), normalize=True, value_range=(-1, 1))
        save_image(image, os.path.join(args.outputs, 'gts', f'{idx}.png'), normalize=True, value_range=(-1, 1))

    # Calculate and print average metrics for each category
    for category, metric in metrics.items():
        if metric['count'] > 0:
            print(f'Category {category}:')
            print(f'  PSNR: {metric["PSNR"] / metric["count"]:.2f}')
            print(f'  SSIM: {metric["SSIM"] / metric["count"]:.4f}')
            print(f'  L1 Error: {metric["L1"] / metric["count"]:.6f}')
        else:
            print(f'Category {category}: No images')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main_worker(args)
