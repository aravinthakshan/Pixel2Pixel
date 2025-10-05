import os
import time
import argparse
import numpy as np
from PIL import Image
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.init as init

import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

import einops


# -------------------------------
parser = argparse.ArgumentParser('Pixel2Pixel with Online Bank Update')
parser.add_argument('--data_path', default='./data', type=str, help='Path to the data')
parser.add_argument('--dataset', default='kodak', type=str, help='Dataset name')
parser.add_argument('--save', default='./results', type=str, help='Directory to save pixel bank results')
parser.add_argument('--out_image', default='./results_image', type=str, help='Directory to save denoised images')
parser.add_argument('--ws', default=40, type=int, help='Window size')
parser.add_argument('--ps', default=7, type=int, help='Patch size')
parser.add_argument('--nn', default=16, type=int, help='Number of nearest neighbors to search')
parser.add_argument('--mm', default=8, type=int, help='Number of pixels in pixel bank to use for training')
parser.add_argument('--nl', default=0.2, type=float, help='Noise level')
parser.add_argument('--nt', default='bernoulli', type=str, help='Noise type: gauss, poiss, saltpepper, bernoulli, impulse')
parser.add_argument('--loss', default='L1', type=str, help='Loss function type')
parser.add_argument('--num_updates', default=3, type=int, help='Number of bank update cycles')
parser.add_argument('--epochs_per_update', default=1000, type=int, help='Epochs between bank updates')
args = parser.parse_args()


# -------------------------------
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda:0"

WINDOW_SIZE = args.ws
PATCH_SIZE = args.ps
NUM_NEIGHBORS = args.nn
noise_level = args.nl
noise_type = args.nt
loss_type = args.loss

transform = transforms.Compose([transforms.ToTensor()])


# -------------------------------
# Function to add noise to an image
# -------------------------------
def add_noise(x, noise_level):
    """Add noise to image tensor (handles device automatically)"""
    if noise_type == 'gauss':
        noisy = x + torch.normal(0, noise_level / 255, x.shape, device=x.device)
        noisy = torch.clamp(noisy, 0, 1)
    elif noise_type == 'poiss':
        noisy = torch.poisson(noise_level * x) / noise_level
    elif noise_type == 'saltpepper':
        prob = torch.rand_like(x)
        noisy = x.clone()
        noisy[prob < noise_level] = 0
        noisy[prob > 1 - noise_level] = 1
    elif noise_type == 'bernoulli':
        prob = torch.rand_like(x)
        mask = (prob > noise_level).float()
        noisy = x * mask
    elif noise_type == 'impulse':
        prob = torch.rand_like(x)
        noise = torch.rand_like(x)
        noisy = x.clone()
        noisy[prob < noise_level] = noise[prob < noise_level]
    else:
        raise ValueError("Unsupported noise type")
    return noisy


# -------------------------------
# Build pixel bank from an image (can be noisy or partially denoised)
# -------------------------------
def build_pixel_bank(img, window_size, patch_size, num_neighbors, loss_type='L1'):
    """
    Build pixel bank from input image
    Args:
        img: [1, C, H, W] image tensor on GPU
        window_size: search window size
        patch_size: patch size for matching
        num_neighbors: number of nearest neighbors
        loss_type: 'L1' or 'L2'
    Returns:
        topk: [H, W, K, C] tensor of nearest neighbor patches
    """
    pad_sz = window_size // 2 + patch_size // 2
    center_offset = window_size // 2
    blk_sz = 64

    # Pad image and extract patches
    img_pad = F.pad(img, (pad_sz, pad_sz, pad_sz, pad_sz), mode='reflect')
    img_unfold = F.unfold(img_pad, kernel_size=patch_size, padding=0, stride=1)
    H_new = img.shape[-2] + window_size
    W_new = img.shape[-1] + window_size
    img_unfold = einops.rearrange(img_unfold, 'b c (h w) -> b c h w', h=H_new, w=W_new)

    num_blk_w = img.shape[-1] // blk_sz
    num_blk_h = img.shape[-2] // blk_sz
    is_window_size_even = (window_size % 2 == 0)
    topk_list = []

    # Iterate over blocks in the image
    for blk_i in range(num_blk_w):
        for blk_j in range(num_blk_h):
            start_h = blk_j * blk_sz
            end_h = (blk_j + 1) * blk_sz + window_size
            start_w = blk_i * blk_sz
            end_w = (blk_i + 1) * blk_sz + window_size

            sub_img_uf = img_unfold[..., start_h:end_h, start_w:end_w]
            sub_img_shape = sub_img_uf.shape

            if is_window_size_even:
                sub_img_uf_inp = sub_img_uf[..., :-1, :-1]
            else:
                sub_img_uf_inp = sub_img_uf

            patch_windows = F.unfold(sub_img_uf_inp, kernel_size=window_size, padding=0, stride=1)
            patch_windows = einops.rearrange(
                patch_windows,
                'b (c k1 k2 k3 k4) (h w) -> b (c k1 k2) (k3 k4) h w',
                k1=patch_size, k2=patch_size, k3=window_size, k4=window_size,
                h=blk_sz, w=blk_sz
            )

            img_center = einops.rearrange(
                sub_img_uf,
                'b (c k1 k2) h w -> b (c k1 k2) 1 h w',
                k1=patch_size, k2=patch_size,
                h=sub_img_shape[-2], w=sub_img_shape[-1]
            )
            img_center = img_center[..., center_offset:center_offset + blk_sz, center_offset:center_offset + blk_sz]

            if loss_type == 'L2':
                distance = torch.sum((img_center - patch_windows) ** 2, dim=1)
            elif loss_type == 'L1':
                distance = torch.sum(torch.abs(img_center - patch_windows), dim=1)
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")

            _, sort_indices = torch.topk(
                distance,
                k=num_neighbors,
                largest=False,
                sorted=True,
                dim=-3
            )

            patch_windows_reshape = einops.rearrange(
                patch_windows,
                'b (c k1 k2) (k3 k4) h w -> b c (k1 k2) (k3 k4) h w',
                k1=patch_size, k2=patch_size, k3=window_size, k4=window_size
            )
            patch_center = patch_windows_reshape[:, :, patch_windows_reshape.shape[2] // 2, ...]
            topk = torch.gather(patch_center, dim=-3,
                                index=sort_indices.unsqueeze(1).repeat(1, 3, 1, 1, 1))
            topk_list.append(topk)

    # Merge the results from all blocks to form the pixel bank
    topk = torch.cat(topk_list, dim=0)
    topk = einops.rearrange(topk, '(w1 w2) c k h w -> k c (w2 h) (w1 w)', w1=num_blk_w, w2=num_blk_h)
    topk = topk.permute(2, 3, 0, 1)  # [H, W, K, C]

    return topk


# -------------------------------
# Network Architecture
# -------------------------------
class Network(nn.Module):
    def __init__(self, n_chan, chan_embed=64):
        super(Network, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv4 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv5 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv6 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)
        self._initialize_weights()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        x = self.conv3(x)
        return torch.sigmoid(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


# -------------------------------
# Loss Functions
# -------------------------------
def mse_loss(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(gt, pred)

loss_f = nn.L1Loss() if args.loss == 'L1' else nn.MSELoss()

def loss_func(img1, img2, model, loss_f):
    pred1 = model(img1)
    loss = loss_f(img2, pred1)
    return loss


# -------------------------------
# Training Function
# -------------------------------
def train(model, optimizer, img_bank):
    N, H, W, C = img_bank.shape
    index1 = torch.randint(0, N, size=(H, W), device=device)
    index1_exp = index1.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img1 = torch.gather(img_bank, 0, index1_exp)
    img1 = img1.permute(0, 3, 1, 2)

    index2 = torch.randint(0, N, size=(H, W), device=device)
    eq_mask = (index2 == index1)
    if eq_mask.any():
        index2[eq_mask] = (index2[eq_mask] + 1) % N
    index2_exp = index2.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img2 = torch.gather(img_bank, 0, index2_exp)
    img2 = img2.permute(0, 3, 1, 2)

    loss = loss_func(img1, img2, model, loss_f)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# -------------------------------
# Testing Function
# -------------------------------
def test(model, noisy_img, clean_img):
    with torch.no_grad():
        pred = torch.clamp(model(noisy_img), 0, 1)
        mse_val = mse_loss(clean_img, pred).item()
        psnr = 10 * np.log10(1 / mse_val)
    return psnr, pred


# -------------------------------
# Main Denoising with Online Bank Update
# -------------------------------
def denoise_images_with_online_update():
    image_folder = os.path.join(args.data_path, args.dataset)
    image_files = sorted(os.listdir(image_folder))

    os.makedirs(args.out_image, exist_ok=True)

    lr = 0.001
    avg_PSNR = 0
    avg_SSIM = 0

    for image_file in image_files:
        print(f"\n{'='*60}")
        print(f"Processing: {image_file}")
        print(f"{'='*60}")
        
        image_path = os.path.join(image_folder, image_file)
        
        # Load clean image
        clean_img = Image.open(image_path)
        clean_img_tensor = transform(clean_img).unsqueeze(0).to(device)
        clean_img_np = io.imread(image_path)
        
        # Generate noisy image
        noisy_img_tensor = add_noise(clean_img_tensor, noise_level)
        
        # Determine bank size based on noise characteristics
        if noise_type == 'gauss' and noise_level == 10 or noise_type == 'bernoulli':
            mm = 2
        elif noise_type == 'gauss' and noise_level == 25:
            mm = 4
        else:
            mm = 8
        
        # Initialize model
        n_chan = clean_img_tensor.shape[1]
        model = Network(n_chan).to(device)
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        
        # ========================================
        # ONLINE BANK UPDATE LOOP
        # ========================================
        total_epochs = args.num_updates * args.epochs_per_update
        current_img = noisy_img_tensor.clone()
        
        for update_cycle in range(args.num_updates):
            print(f"\n--- Bank Update Cycle {update_cycle + 1}/{args.num_updates} ---")
            
            # Build/Rebuild pixel bank from current image state
            start_bank_time = time.time()
            print(f"Building pixel bank from {'noisy' if update_cycle == 0 else 'partially denoised'} image...")
            
            img_bank = build_pixel_bank(
                current_img, 
                WINDOW_SIZE, 
                PATCH_SIZE, 
                NUM_NEIGHBORS, 
                args.loss
            )
            
            # Select top mm patches
            img_bank = img_bank.permute(2, 3, 0, 1)  # [K, C, H, W]
            img_bank = img_bank[:mm]
            
            bank_time = time.time() - start_bank_time
            print(f"Pixel bank built in {bank_time:.2f}s. Shape: {img_bank.shape}")
            
            # Learning rate schedule for this cycle
            milestones = [
                args.epochs_per_update // 3,
                2 * args.epochs_per_update // 3,
                5 * args.epochs_per_update // 6
            ]
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
            
            # Train for epochs_per_update epochs
            print(f"Training for {args.epochs_per_update} epochs...")
            train_start = time.time()
            
            for epoch in range(args.epochs_per_update):
                loss = train(model, optimizer, img_bank)
                scheduler.step()
                
                if (epoch + 1) % 200 == 0:
                    global_epoch = update_cycle * args.epochs_per_update + epoch + 1
                    print(f"  Epoch [{global_epoch}/{total_epochs}] Loss: {loss:.6f}")
            
            train_time = time.time() - train_start
            print(f"Training completed in {train_time:.2f}s")
            
            # Test current performance
            with torch.no_grad():
                current_img = torch.clamp(model(noisy_img_tensor), 0, 1)
                mse_val = mse_loss(clean_img_tensor, current_img).item()
                psnr = 10 * np.log10(1 / mse_val)
                print(f"Intermediate PSNR: {psnr:.2f} dB")
            
            # For next cycle, use the denoised image to build better bank
            # (except on last iteration where we just test)
        
        # ========================================
        # Final Evaluation
        # ========================================
        print(f"\n--- Final Evaluation ---")
        PSNR, out_img = test(model, noisy_img_tensor, clean_img_tensor)
        
        # Save denoised image
        out_img_pil = to_pil_image(out_img.squeeze(0).cpu())
        out_img_save_path = os.path.join(args.out_image, os.path.splitext(image_file)[0] + '.png')
        out_img_pil.save(out_img_save_path)
        
        # Save noisy image for comparison
        noisy_img_pil = to_pil_image(noisy_img_tensor.squeeze(0).cpu())
        noisy_img_save_path = os.path.join(args.out_image, os.path.splitext(image_file)[0] + '_noisy.png')
        noisy_img_pil.save(noisy_img_save_path)
        
        # Compute SSIM
        out_img_loaded = io.imread(out_img_save_path)
        SSIM, _ = compare_ssim(clean_img_np, out_img_loaded, full=True, multichannel=True)
        
        print(f"\nFinal Results for {image_file}:")
        print(f"  PSNR: {PSNR:.2f} dB")
        print(f"  SSIM: {SSIM:.4f}")
        
        avg_PSNR += PSNR
        avg_SSIM += SSIM

    # Print average results
    avg_PSNR /= len(image_files)
    avg_SSIM /= len(image_files)
    print(f"\n{'='*60}")
    print(f"Average Results Across Dataset:")
    print(f"  Average PSNR: {avg_PSNR:.2f} dB")
    print(f"  Average SSIM: {avg_SSIM:.4f}")
    print(f"{'='*60}")


# -------------------------------
if __name__ == "__main__":
    print("="*60)
    print("Online Pixel Bank Update Denoising")
    print(f"Configuration:")
    print(f"  Noise type: {args.nt}, level: {args.nl}")
    print(f"  Window size: {args.ws}, Patch size: {args.ps}")
    print(f"  Number of neighbors: {args.nn}")
    print(f"  Bank updates: {args.num_updates}")
    print(f"  Epochs per update: {args.epochs_per_update}")
    print(f"  Total epochs: {args.num_updates * args.epochs_per_update}")
    print("="*60)
    
    denoise_images_with_online_update()
