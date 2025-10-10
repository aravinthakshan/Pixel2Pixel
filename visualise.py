import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser('Pixel2Pixel')
parser.add_argument('--image_name', default='kodim01', type=str)
parser.add_argument('--bank_dir', default='./results/', type=str)
args = parser.parse_args()

bank_path = os.path.join(args.bank_dir, args.image_name)
device = 'cpu'

img_bank_arr = np.load(bank_path + '.npy')

if img_bank_arr.ndim == 3:
    img_bank_arr = np.expand_dims(img_bank_arr, axis=1)

img_bank = img_bank_arr.astype(np.float32).transpose((2, 0, 1, 3))
img_bank = torch.from_numpy(img_bank).to(device)

# Create output directory
out_dir = os.path.join('visualised', args.image_name)
os.makedirs(out_dir, exist_ok=True)

# Save all banks
for idx in range(img_bank.shape[0]):
    img = img_bank[idx].cpu().numpy()
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Bank {idx}')
    plt.savefig(os.path.join(out_dir, f'{args.image_name}_{idx}.png'), bbox_inches='tight', pad_inches=0)
    plt.close()
