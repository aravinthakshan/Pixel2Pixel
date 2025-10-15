import os
import time
import argparse
import numpy as np
from PIL import Image
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim

from src.loss import loss_func, mse_loss
from src.utils import parse_iteration_params
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from network import NetworkSyn, NetworkReal
from src.bank_creation import *

import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

# -------------------------------
parser = argparse.ArgumentParser('Pixel2Pixel')
# Paths
parser.add_argument('--data_path', default='./data', type=str, help='Path to the data')
parser.add_argument('--bank_dir', default='./results', type=str, help='Directory to save pixel bank results')
parser.add_argument('--out_image', default='./results_image', type=str, help='Directory to save denoised images')

parser.add_argument('--dataset', default='kodak', type=str, help='Dataset name')
parser.add_argument('--device', default='cuda:0', type=str, help='Device type - CUDA or CPU')
parser.add_argument('--loss', default='L1', type=str, help='Loss function type')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of bank reconstruction iterations')
parser.add_argument('--epochs_per_iter', default=1000, type=int, help='Epochs per iteration')

#Model HPs
parser.add_argument('--ws', default=40, type=int, help='Window size')
parser.add_argument('--ps', default=7, type=int, help='Patch size')
parser.add_argument('--nn', default=16, type=int, help='Number of nearest neighbors to search')
parser.add_argument('--mm', default=8, type=int, help='Number of pixels in pixel bank to use for training')

# Arguments for syn datasets
parser.add_argument('--nl', default=0.2, type=float, help='Noise level, for saltpepper and impulse noise, enter half the noise level.')
parser.add_argument('--nt', default='bernoulli', type=str, help='Noise type: gauss, poiss, saltpepper, bernoulli, impulse')
parser.add_argument('--use_quality_weights', default=False, type=bool, help='Use quality-based sampling weights')
parser.add_argument('--alpha', default=2.0, type=float, help='Sharpness of quality scoring (higher = more selective)')

# Arguements for real datasets
parser.add_argument('--gt_dir', default='GT', type=str, help='Folder name for ground truth images')
parser.add_argument('--noisy_dir', default='Noisy', type=str, help='Folder name for noisy images')

# Progressive growing parameters
parser.add_argument('--nn_layers', default='6,9,12', type=str, 
                   help='Comma-separated number of conv layers per iteration (e.g., "6,9,12")')

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

transform = transforms.Compose([transforms.ToTensor()])

args = parser.parse_args()


def train_real(model, optimizer, img_bank, args):
    N, H, W, C = img_bank.shape
    device = args.device
    loss_f = nn.L1Loss() if args.loss == 'L1' else nn.MSELoss()

    index1 = torch.randint(0, N, size=(H, W), device=device)
    index1_exp = index1.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img1 = torch.gather(img_bank, 0, index1_exp)  # Shape: (1, H, W, C)
    img1 = img1.permute(0, 3, 1, 2)  # (1, C, H, W)

    index2 = torch.randint(0, N, size=(H, W), device=device)
    eq_mask = (index2 == index1)
    if eq_mask.any():
        index2[eq_mask] = (index2[eq_mask] + 1) % N
    index2_exp = index2.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img2 = torch.gather(img_bank, 0, index2_exp)
    img2 = img2.permute(0, 3, 1, 2)

    loss = loss_func(model, img1, img2, loss_f)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def train_syn(model, optimizer, img_bank, args, quality_weights=None):
    N, H, W, C = img_bank.shape

    device = args.device
    loss_f = nn.L1Loss() if args.loss == 'L1' else nn.MSELoss()

    if args.use_quality_weights and quality_weights is not None:
        # Sample based on quality weights
        flat_weights = quality_weights.view(-1, N)
        
        index1 = torch.multinomial(flat_weights, num_samples=1, replacement=True).view(H, W)
        index2 = torch.multinomial(flat_weights, num_samples=1, replacement=True).view(H, W)
        
        # Ensure different indices
        eq_mask = (index2 == index1)
        if eq_mask.any():
            eq_positions = eq_mask.view(-1)
            eq_weights = flat_weights[eq_positions]
            new_samples = torch.multinomial(eq_weights, num_samples=1, replacement=True)
            index2.view(-1)[eq_positions] = new_samples.squeeze()
            eq_mask = (index2 == index1)
            if eq_mask.any():
                index2[eq_mask] = (index2[eq_mask] + 1) % N
    else:
        # Original uniform sampling
        index1 = torch.randint(0, N, size=(H, W), device=device)
        index2 = torch.randint(0, N, size=(H, W), device=device)
        eq_mask = (index2 == index1)
        if eq_mask.any():
            index2[eq_mask] = (index2[eq_mask] + 1) % N
    
    index1_exp = index1.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img1 = torch.gather(img_bank, 0, index1_exp)
    img1 = img1.permute(0, 3, 1, 2)

    index2_exp = index2.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img2 = torch.gather(img_bank, 0, index2_exp)
    img2 = img2.permute(0, 3, 1, 2)

    loss = loss_func(model, img1, img2, loss_f)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, noisy_img, clean_img):
    with torch.no_grad():
        pred = torch.clamp(model(noisy_img), 0, 1)
        mse_val = mse_loss(clean_img, pred).item()
        psnr = 10 * np.log10(1 / mse_val)
    return psnr, pred

def denoise_real(args):
    device = args.device
    # The pixel bank directory should match the one used in construction
    bank_dir = os.path.join(args.bank_dir, args.dataset, '_'.join(str(i) for i in [args.ws, args.ps, args.nn, args.loss]))
    gt_folder = os.path.join(args.data_path, args.dataset, args.gt_dir)
    gt_files = sorted(os.listdir(gt_folder))

    os.makedirs(args.out_image, exist_ok=True)

    max_epoch = 3000
    lr = 0.001
    avg_PSNR = 0
    avg_SSIM = 0

    for image_file in gt_files:
        image_path = os.path.join(gt_folder, image_file)
        clean_img = Image.open(image_path)
        clean_img_tensor = transform(clean_img).unsqueeze(0).to(device)
        clean_img_np = io.imread(image_path)

        bank_path = os.path.join(bank_dir, os.path.splitext(image_file)[0])
        if not os.path.exists(bank_path + '.npy'):
            print(f"Pixel bank for {image_file} not found, skipping denoising.")
            continue

        img_bank_arr = np.load(bank_path + '.npy')
        if img_bank_arr.ndim == 3:
            img_bank_arr = np.expand_dims(img_bank_arr, axis=1)
        # Transpose to (k, H, W, c)
        img_bank = img_bank_arr.astype(np.float32).transpose((2, 0, 1, 3))
        # Use only the first mm banks for training
        img_bank = img_bank[:args.mm]
        img_bank = torch.from_numpy(img_bank).to(device)

        # Use the first bank as the noisy input (reshaped to (1, C, H, W))
        noisy_img = img_bank[0].unsqueeze(0).permute(0, 3, 1, 2)

        n_chan = clean_img_tensor.shape[1]
        model = NetworkReal(n_chan).to(device)
        print(f"Number of parameters for {image_file}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=[1500, 2000, 2500], gamma=0.5)

        for epoch in range(max_epoch):
            train_real(model, optimizer, img_bank, args)
            scheduler.step()

        PSNR, out_img = test(model, noisy_img, clean_img_tensor)
        out_img_pil = to_pil_image(out_img.squeeze(0))
        out_img_save_path = os.path.join(args.out_image, os.path.splitext(image_file)[0] + '.png')
        out_img_pil.save(out_img_save_path)

        noisy_img_pil = to_pil_image(noisy_img.squeeze(0))
        noisy_img_save_path = os.path.join(args.out_image, os.path.splitext(image_file)[0] + '_noisy.png')
        noisy_img_pil.save(noisy_img_save_path)

        out_img_loaded = io.imread(out_img_save_path)
        min_dim = min(clean_img_np.shape[0], clean_img_np.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        SSIM, _ = compare_ssim(clean_img_np, out_img_loaded, full=True, channel_axis=2, win_size=win_size)
        print(f"Image: {image_file} | PSNR: {PSNR:.2f} dB | SSIM: {SSIM:.4f}")
        avg_PSNR += PSNR
        avg_SSIM += SSIM

    avg_PSNR /= len(gt_files)
    avg_SSIM /= len(gt_files)
    print(f"Average PSNR: {avg_PSNR:.2f} dB, Average SSIM: {avg_SSIM:.4f}")


def denoise_syn(args):
    noise_type = args.nt
    noise_level = args.nl
    device = args.device

    bank_dir = os.path.join(args.bank_dir, '_'.join(
        str(i) for i in [args.dataset, args.nt, args.nl, args.ws, args.ps, args.nn, args.loss]))
    image_folder = os.path.join(args.data_path, args.dataset)
    image_files = sorted(os.listdir(image_folder))

    os.makedirs(args.out_image, exist_ok=True)

    # Parse progressive parameters
    nn_layers_list = parse_iteration_params(args)

    lr = 0.001
    avg_PSNR = 0
    avg_SSIM = 0

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        clean_img = Image.open(image_path)
        clean_img_tensor = transform(clean_img).unsqueeze(0).to(device)
        clean_img_np = io.imread(image_path)

        file_name_without_ext = os.path.splitext(image_file)[0]
        bank_path = os.path.join(bank_dir, file_name_without_ext)
        if not os.path.exists(bank_path + '.npy'):
            print(f"Pixel bank for {image_file} not found, skipping denoising.")
            continue

        # Determine mm parameter
        if noise_type=='gauss' and noise_level==10 or noise_type=='bernoulli':
            args.mm=2
        elif noise_type=='gauss' and noise_level==25:
            args.mm = 4
        else:
            args.mm = 8

        n_chan = clean_img_tensor.shape[1]
        
        # Iterative bank reconstruction
        for iteration in range(args.num_iterations):
            print(f"\n{'='*60}")
            print(f"Image: {image_file} | Iteration {iteration + 1}/{args.num_iterations}")
            
            # Get parameters for this iteration
            num_layers = nn_layers_list[iteration]
            
            # Create network with specified number of layers
            model = NetworkSyn(n_chan, num_conv_layers=num_layers).to(device)
            
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            
            # Load current pixel bank
            img_bank_arr = np.load(bank_path + '.npy')
            if img_bank_arr.ndim == 3:
                img_bank_arr = np.expand_dims(img_bank_arr, axis=1)
            img_bank = img_bank_arr.astype(np.float32).transpose((2, 0, 1, 3))
            img_bank = img_bank[:args.mm]
            img_bank = torch.from_numpy(img_bank).to(device)

            # Load distances and compute quality weights
            quality_weights = None
            if args.use_quality_weights:
                dist_path = os.path.join(bank_dir, file_name_without_ext + '_distances.npy')
                if os.path.exists(dist_path):
                    distances_arr = np.load(dist_path)
                    distances = torch.from_numpy(distances_arr.astype(np.float32)).to(device)
                    distances = distances[..., :args.mm]
                    
                    quality_weights = compute_quality_weights_distance(
                            distances, 
                            alpha=args.alpha
                        )
                else:
                    print("Distance file not found, using uniform sampling")

            noisy_img = img_bank[0].unsqueeze(0).permute(0, 3, 1, 2)

            # Reset scheduler for each iteration
            scheduler = MultiStepLR(optimizer, 
                                milestones=[int(args.epochs_per_iter*0.5), 
                                            int(args.epochs_per_iter*0.67), 
                                            int(args.epochs_per_iter*0.83)], 
                                gamma=0.5)

            # Train for epochs_per_iter
            current_mse = 0
            epoch = 0
            while(epoch <(args.epochs_per_iter)):
                train_syn(model, optimizer, img_bank, args, quality_weights)
                scheduler.step()
                
                if (epoch + 1) % 3000 == 0:
                    with torch.no_grad():
                        current_pred = torch.clamp(model(noisy_img), 0, 1)
                        prev_mse = current_mse
                        current_mse = mse_loss(clean_img_tensor, current_pred).item()

                        #Sigmoid can cause the weights to move to 0, Removing final sigmoid layer if that happens
                        if (current_mse-prev_mse==0.0):
                            print("Restarting trianing with sigmoid turned off")
                            model.use_sigmoid = False
                            epoch=0
                        current_psnr = 10 * np.log10(1 / current_mse)
                
                epoch+=1

            # Get partially denoised image
            with torch.no_grad():
                denoised_img = torch.clamp(model(noisy_img), 0, 1)
                current_mse = mse_loss(clean_img_tensor, denoised_img).item()
                current_psnr = 10 * np.log10(1 / current_mse)
            
            print(f"Iteration {iteration + 1} complete - PSNR: {current_psnr:.2f} dB")
            
            # Rebuild pixel bank using partially denoised image (except for last iteration)
            if iteration < args.num_iterations - 1:
                print(f"Rebuilding pixel bank with partially denoised image...")
                start_time = time.time()
                topk, distances = construct_pixel_bank_from_image(denoised_img, file_name_without_ext, bank_dir, args)
                elapsed = time.time() - start_time
                print(f"Bank rebuilt in {elapsed:.2f} seconds. Shape: {topk.shape}")
            
        # Final evaluation
        PSNR, out_img = test(model, noisy_img, clean_img_tensor)
        out_img_pil = to_pil_image(out_img.squeeze(0))
        out_img_save_path = os.path.join(args.out_image, os.path.splitext(image_file)[0] + '.png')
        out_img_pil.save(out_img_save_path)

        noisy_img_pil = to_pil_image(noisy_img.squeeze(0))
        noisy_img_save_path = os.path.join(args.out_image, os.path.splitext(image_file)[0] + '_noisy.png')
        noisy_img_pil.save(noisy_img_save_path)

        out_img_loaded = io.imread(out_img_save_path)
        min_dim = min(clean_img_np.shape[0], clean_img_np.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        SSIM, _ = compare_ssim(clean_img_np, out_img_loaded, full=True, channel_axis=2, win_size=win_size)
        print(f"\nFinal Results - Image: {image_file} | PSNR: {PSNR:.2f} dB | SSIM: {SSIM:.4f}")
        avg_PSNR += PSNR
        avg_SSIM += SSIM
        model.use_sigmoid = True

    avg_PSNR /= len(image_files)
    avg_SSIM /= len(image_files)
    print(f"\n{'='*60}")
    print(f"Average PSNR: {avg_PSNR:.2f} dB, Average SSIM: {avg_SSIM:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset}")
    
    if args.dataset in ['kodak', 'mcmaster']:
        print(f"  Noise type: {args.nt}, level: {args.nl}")
        print(f"  Iterations: {args.num_iterations}, Epochs/iter: {args.epochs_per_iter}")
        
        nn_layers_list = parse_iteration_params(args)
        print(f"  Network layers per iteration: {nn_layers_list}")
        
        if args.use_quality_weights:
            print(f"  Quality weighting: ENABLED")
            print(f"  Sampling strategy: Distance-based")
            print(f"  Distance alpha: {args.alpha}")
        else:
            print(f"  Quality weighting: DISABLED (uniform sampling)")
        
        print("="*60)
        print("\nConstructing initial pixel banks from noisy images...")
        construct_pixel_bank(args)
        print("\nStarting iterative denoising...")
        denoise_syn(args)
    
    if args.dataset in ['sidd', 'polyu']:
        print("Constructing pixel banks from noisy images ...")
        construct_pixel_bank_real(args)
        print("Starting denoising using constructed pixel banks ...")
        denoise_real(args)