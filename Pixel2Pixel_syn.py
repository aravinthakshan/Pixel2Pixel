#!/usr/bin/env python3
# Pixel2Pixel_syn.py
# Single-file Pixel2Pixel + Optuna script (modern AMP; no deprecated torch.cuda.amp usage)

import os
import time
import argparse
import json
import numpy as np
from PIL import Image
from skimage import io
from skimage.metrics import structural_similarity as compare_ssim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.init as init

import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

import einops
import optuna
from optuna.trial import TrialState

# ========================================================================
# CONFIGURATION SECTION - EASY TO MODIFY
# ========================================================================

OPTUNA_CONFIG = {
    'noise_configs': [
        ('gauss', 10),
        ('gauss', 25),
        ('gauss', 30),
        ('poiss', 10),
        ('poiss', 25),
        ('poiss', 50),
    ],
    'search_space': {
        'layer_schedule': ['3,5,6', '6,9,12'],
        'num_iterations': [1, 3],
        'epochs_per_iter': [1000, 3000, 4000, 5000],
        'chan_embed': [64, 96],
    },
    'n_trials': 100,
    'timeout': None,
    'n_jobs': 1,
}

GPU_CONFIG = {
    'use_amp': True,
    'batch_accumulation': 2,
    'pin_memory': True,
    'num_workers': 4,
    'prefetch_factor': 2,
}

FIXED_PARAMS = {
    'ws': 40,
    'ps': 7,
    'nn': 16,
    'mm': 8,
    'loss': 'L1',
    'alpha': 2.0,
    'lr': 0.001,
    'use_quality_weights': True,
    'progressive_growing': True,
}

# ========================================================================
# ARGUMENT PARSER
# ========================================================================

parser = argparse.ArgumentParser('Pixel2Pixel with Optuna Optimization')
parser.add_argument('--data_path', default='./data', type=str)
parser.add_argument('--dataset', default='kodak', type=str)
parser.add_argument('--save', default='./results', type=str)
parser.add_argument('--out_image', default='./results_image', type=str)
parser.add_argument('--mode', default='optuna', type=str, choices=['optuna', 'single'])
parser.add_argument('--study_name', default='pixel2pixel_study', type=str)
parser.add_argument('--storage', default='sqlite:///optuna_study.db', type=str)

# single-run overrides
parser.add_argument('--ws', default=40, type=int)
parser.add_argument('--ps', default=7, type=int)
parser.add_argument('--nn', default=16, type=int)
parser.add_argument('--mm', default=8, type=int)
parser.add_argument('--nl', default=0.2, type=float)
parser.add_argument('--nt', default='gauss', type=str)
parser.add_argument('--loss', default='L1', type=str)
parser.add_argument('--num_iterations', default=3, type=int)
parser.add_argument('--epochs_per_iter', default=1000, type=int)
parser.add_argument('--use_quality_weights', default=True, type=bool)
parser.add_argument('--alpha', default=2.0, type=float)
parser.add_argument('--progressive_growing', default=True, type=bool)
parser.add_argument('--layer_schedule', default='3,5,6', type=str)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--chan_embed', default=64, type=int)

args = parser.parse_args()

# ========================================================================
# SETUP
# ========================================================================

torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
# allow TF32 for speed on Ampere+ (unchanged)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_type_for_autocast = 'cuda' if device.type == 'cuda' else 'cpu'
print(f"Using device: {device}")
if device.type == 'cuda':
    try:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except Exception:
        pass

transform = transforms.Compose([transforms.ToTensor()])

# ========================================================================
# NOISE FUNCTIONS
# ========================================================================

def add_noise(x, noise_level, noise_type):
    if noise_type == 'gauss':
        noisy = x + torch.normal(0, noise_level / 255.0, x.shape, device=x.device)
        noisy = torch.clamp(noisy, 0, 1)
    elif noise_type == 'poiss':
        noisy = torch.poisson(noise_level * x) / float(noise_level)
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

# ========================================================================
# PIXEL BANK CONSTRUCTION (GPU-OPTIMIZED, with modern autocast)
# ========================================================================

def construct_pixel_bank_from_image(img_tensor, file_name_without_ext, bank_dir, params):
    """
    Build and save the pixel bank and distances for one image tensor.
    img_tensor should be shaped (1, C, H, W) and on the target device.
    """
    # use autocast context for mixed precision (if enabled)
    with torch.autocast(device_type=device_type_for_autocast, enabled=GPU_CONFIG['use_amp'] and device.type == 'cuda'):
        pad_sz = params['ws'] // 2 + params['ps'] // 2
        center_offset = params['ws'] // 2
        blk_sz = 128

        img = img_tensor  # shape 1,C,H,W

        img_pad = F.pad(img, (pad_sz, pad_sz, pad_sz, pad_sz), mode='reflect')
        img_unfold = F.unfold(img_pad, kernel_size=params['ps'], padding=0, stride=1)
        H_new = img.shape[-2] + params['ws']
        W_new = img.shape[-1] + params['ws']
        img_unfold = einops.rearrange(img_unfold, 'b c (h w) -> b c h w', h=H_new, w=W_new)

        num_blk_w = max(1, img.shape[-1] // blk_sz)
        num_blk_h = max(1, img.shape[-2] // blk_sz)
        is_window_size_even = (params['ws'] % 2 == 0)

        topk_list = []
        distance_list = []

        for blk_i in range(num_blk_w):
            for blk_j in range(num_blk_h):
                start_h = blk_j * blk_sz
                end_h = (blk_j + 1) * blk_sz + params['ws']
                start_w = blk_i * blk_sz
                end_w = (blk_i + 1) * blk_sz + params['ws']

                sub_img_uf = img_unfold[..., start_h:end_h, start_w:end_w]
                sub_img_shape = sub_img_uf.shape

                if is_window_size_even:
                    sub_img_uf_inp = sub_img_uf[..., :-1, :-1]
                else:
                    sub_img_uf_inp = sub_img_uf

                patch_windows = F.unfold(sub_img_uf_inp, kernel_size=params['ws'], padding=0, stride=1)
                # patch_windows: b (c * k1 * k2 * k3 * k4) (h*w)
                patch_windows = einops.rearrange(
                    patch_windows,
                    'b (c k1 k2 k3 k4) (h w) -> b (c k1 k2) (k3 k4) h w',
                    k1=params['ps'], k2=params['ps'], k3=params['ws'], k4=params['ws'],
                    h=blk_sz, w=blk_sz
                )

                img_center = einops.rearrange(
                    sub_img_uf,
                    'b (c k1 k2) h w -> b (c k1 k2) 1 h w',
                    k1=params['ps'], k2=params['ps'],
                    h=sub_img_shape[-2], w=sub_img_shape[-1]
                )
                img_center = img_center[..., center_offset:center_offset + blk_sz, center_offset:center_offset + blk_sz]

                if params['loss'] == 'L2':
                    distance = torch.sum((img_center - patch_windows) ** 2, dim=1)
                else:
                    distance = torch.sum(torch.abs(img_center - patch_windows), dim=1)

                topk_distances, sort_indices = torch.topk(
                    distance, k=params['nn'], largest=False, sorted=True, dim=-3
                )

                patch_windows_reshape = einops.rearrange(
                    patch_windows,
                    'b (c k1 k2) (k3 k4) h w -> b c (k1 k2) (k3 k4) h w',
                    k1=params['ps'], k2=params['ps'], k3=params['ws'], k4=params['ws']
                )
                patch_center = patch_windows_reshape[:, :, patch_windows_reshape.shape[2] // 2, ...]
                # gather topk centers along the patch index dimension
                topk = torch.gather(patch_center, dim=-3,
                                    index=sort_indices.unsqueeze(1).repeat(1, patch_center.shape[1], 1, 1, 1))
                topk_list.append(topk)
                distance_list.append(topk_distances)

        if len(topk_list) == 0:
            # fallback: nothing found (tiny image), produce empty arrays
            topk = torch.empty((0, img.shape[1], 0, 0), device=img.device)
            distances = torch.empty((0, 0, 0), device=img.device)
        else:
            topk = torch.cat(topk_list, dim=0)
            # rearrange to k c (w2*h) (w1*w)
            topk = einops.rearrange(topk, '(w1 w2) c k h w -> k c (w2 h) (w1 w)', w1=num_blk_w, w2=num_blk_h)
            topk = topk.permute(2, 3, 0, 1)

            distances = torch.cat(distance_list, dim=0)
            distances = einops.rearrange(distances, '(w1 w2) k h w -> k (w2 h) (w1 w)', w1=num_blk_w, w2=num_blk_h)
            distances = distances.permute(1, 2, 0)

        # save numpy arrays (move to cpu)
        np.save(os.path.join(bank_dir, file_name_without_ext), topk.cpu().numpy())
        np.save(os.path.join(bank_dir, file_name_without_ext + '_distances'), distances.cpu().numpy())

    return topk, distances


def construct_pixel_bank(params):
    bank_dir = os.path.join(args.save, '_'.join(
        str(i) for i in [args.dataset, params['nt'], params['nl'], params['ws'], params['ps'], params['nn'], params['loss']]))
    os.makedirs(bank_dir, exist_ok=True)

    image_folder = os.path.join(args.data_path, args.dataset)
    image_files = sorted(os.listdir(image_folder))

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        start_time = time.time()

        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0)  # 1,C,H,W
        img = add_noise(img, params['nl'], params['nt']).squeeze(0)
        img = img.to(device)[None, ...]

        file_name_without_ext = os.path.splitext(image_file)[0]
        topk, distances = construct_pixel_bank_from_image(img, file_name_without_ext, bank_dir, params)

        elapsed = time.time() - start_time
        print(f"Processed {image_file} in {elapsed:.2f}s. Bank shape: {topk.shape}")

    print("Pixel bank construction completed.")
    return bank_dir

# ========================================================================
# NETWORK
# ========================================================================

class Network(nn.Module):
    def __init__(self, n_chan, chan_embed=64, max_layers=12):
        super(Network, self).__init__()
        self.max_layers = max_layers
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.middle_layers = nn.ModuleList([
            nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
            for _ in range(max_layers - 1)
        ])
        self.conv_out = nn.Conv2d(chan_embed, n_chan, 1)
        self.active_layers = max_layers

        self._initialize_weights()

    def forward(self, x):
        x = self.act(self.conv1(x))
        num_middle = self.active_layers - 2
        for i in range(min(num_middle, len(self.middle_layers))):
            x = self.act(self.middle_layers[i](x))
        x = self.conv_out(x)
        return torch.sigmoid(x)

    def set_active_layers(self, num_layers):
        self.active_layers = max(2, min(self.max_layers, num_layers))

    def get_active_parameters(self):
        params = []
        params.extend(list(self.conv1.parameters()))
        params.extend(list(self.conv_out.parameters()))
        num_middle = self.active_layers - 2
        for i in range(min(num_middle, len(self.middle_layers))):
            params.extend(list(self.middle_layers[i].parameters()))
        return params

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

# ========================================================================
# QUALITY WEIGHTS & TRAINING
# ========================================================================

def compute_quality_weights(distances, alpha=2.0):
    dist_min = distances.min(dim=-1, keepdim=True)[0]
    dist_max = distances.max(dim=-1, keepdim=True)[0]
    dist_range = torch.clamp(dist_max - dist_min, min=1e-8)
    normalized_dist = (distances - dist_min) / dist_range

    optimal_distance = 0.5
    quality_scores = torch.exp(-alpha * (normalized_dist - optimal_distance) ** 2)
    too_similar_penalty = torch.exp(-10 * normalized_dist)
    quality_scores = quality_scores * (1 - 0.5 * too_similar_penalty)

    weights = quality_scores / (quality_scores.sum(dim=-1, keepdim=True) + 1e-8)
    return weights


def train_step(model, optimizer, img_bank, quality_weights, params):
    """
    Single training step (no GradScaler; using native autocast for forward).
    img_bank: Tensor shaped (N, H, W, C)
    Returns scalar loss float.
    """
    N, H, W, C = img_bank.shape

    if params['use_quality_weights'] and quality_weights is not None:
        flat_weights = quality_weights.view(-1, N)
        index1 = torch.multinomial(flat_weights, num_samples=1, replacement=True).view(H, W)
        index2 = torch.multinomial(flat_weights, num_samples=1, replacement=True).view(H, W)
        eq_mask = (index2 == index1)
        if eq_mask.any():
            index2[eq_mask] = (index2[eq_mask] + 1) % N
    else:
        index1 = torch.randint(0, N, size=(H, W), device=device)
        index2 = torch.randint(0, N, size=(H, W), device=device)
        eq_mask = (index2 == index1)
        if eq_mask.any():
            index2[eq_mask] = (index2[eq_mask] + 1) % N

    index1_exp = index1.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img1 = torch.gather(img_bank, 0, index1_exp).permute(0, 3, 1, 2)  # 1, C, H, W

    index2_exp = index2.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img2 = torch.gather(img_bank, 0, index2_exp).permute(0, 3, 1, 2)

    loss_f = nn.L1Loss() if params['loss'] == 'L1' else nn.MSELoss()

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device_type_for_autocast, enabled=GPU_CONFIG['use_amp'] and device.type == 'cuda'):
        pred = model(img1)
        loss = loss_f(img2, pred)

    loss.backward()
    optimizer.step()

    return float(loss.item())


def test(model, noisy_img, clean_img):
    with torch.no_grad():
        with torch.autocast(device_type=device_type_for_autocast, enabled=GPU_CONFIG['use_amp'] and device.type == 'cuda'):
            pred = torch.clamp(model(noisy_img), 0, 1)
        mse_val = F.mse_loss(clean_img, pred).item()
        psnr = 10 * np.log10(1.0 / mse_val) if mse_val > 0 else 100.0
    return psnr, pred

# ========================================================================
# MAIN DENOISING FUNCTION
# ========================================================================

def denoise_images(params, verbose=True):
    bank_dir = os.path.join(args.save, '_'.join(
        str(i) for i in [args.dataset, params['nt'], params['nl'], params['ws'], params['ps'], params['nn'], params['loss']]))
    os.makedirs(args.save, exist_ok=True)

    # Check if bank exists
    bank_exists = True
    image_folder = os.path.join(args.data_path, args.dataset)
    image_files = sorted(os.listdir(image_folder))

    for image_file in image_files:
        file_name = os.path.splitext(image_file)[0]
        if not os.path.exists(os.path.join(bank_dir, file_name + '.npy')):
            bank_exists = False
            break

    if not bank_exists:
        if verbose:
            print("Pixel bank not found, constructing...")
        bank_dir = construct_pixel_bank(params)

    os.makedirs(args.out_image, exist_ok=True)

    avg_PSNR = 0.0
    avg_SSIM = 0.0
    num_images = 0

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        clean_img = Image.open(image_path).convert('RGB')
        clean_img_tensor = transform(clean_img).unsqueeze(0).to(device)
        clean_img_np = io.imread(image_path)

        file_name_without_ext = os.path.splitext(image_file)[0]
        bank_path = os.path.join(bank_dir, file_name_without_ext)

        if not os.path.exists(bank_path + '.npy'):
            continue

        # Adjust mm based on noise
        if (params['nt'] == 'gauss' and params['nl'] == 10) or params['nt'] == 'bernoulli':
            mm = 2
        elif params['nt'] == 'gauss' and params['nl'] == 25:
            mm = 4
        else:
            mm = params.get('mm', FIXED_PARAMS['mm'])

        n_chan = clean_img_tensor.shape[1]

        # Parse layer schedule
        layer_schedule_config = [int(x.strip()) for x in str(params['layer_schedule']).split(',')]
        max_layers_needed = max(layer_schedule_config)

        model = Network(n_chan, chan_embed=params['chan_embed'], max_layers=max_layers_needed).to(device)
        model = torch.compile(model)
        
        # Generate layer schedule
        num_iters = int(params['num_iterations'])
        if len(layer_schedule_config) == num_iters:
            layer_schedule = layer_schedule_config
        elif len(layer_schedule_config) == 1:
            layer_schedule = [layer_schedule_config[0]] * num_iters
        else:
            layer_schedule = []
            for i in range(num_iters):
                idx = i % len(layer_schedule_config)
                layer_schedule.append(layer_schedule_config[idx])

        # Apply progressive growing (if requested and config len==1)
        if params.get('progressive_growing', False) and len(layer_schedule_config) == 1:
            base_layers = layer_schedule_config[0]
            if num_iters == 1:
                layer_schedule = [base_layers]
            elif num_iters > 1:
                min_layers = max(2, base_layers - num_iters + 1)
                layer_schedule = []
                for i in range(num_iters):
                    progress = i / (num_iters - 1)
                    layers = int(min_layers + progress * (base_layers - min_layers))
                    layer_schedule.append(layers)

        # Iterative training
        for iteration in range(num_iters):
            current_layers = layer_schedule[iteration]
            model.set_active_layers(current_layers)

            if params.get('progressive_growing', False):
                active_params = model.get_active_parameters()
                optimizer = optim.AdamW(active_params, lr=params['lr'])
            else:
                optimizer = optim.AdamW(model.parameters(), lr=params['lr'])

            # Load bank
            img_bank_arr = np.load(bank_path + '.npy')
            if img_bank_arr.ndim == 3:
                img_bank_arr = np.expand_dims(img_bank_arr, axis=1)
            img_bank = img_bank_arr.astype(np.float32).transpose((2, 0, 1, 3))
            img_bank = img_bank[:mm]
            img_bank = torch.from_numpy(img_bank).to(device, non_blocking=True)

            # Quality weights
            quality_weights = None
            if params.get('use_quality_weights', False):
                dist_path = os.path.join(bank_dir, file_name_without_ext + '_distances.npy')
                if os.path.exists(dist_path):
                    distances_arr = np.load(dist_path)
                    distances = torch.from_numpy(distances_arr.astype(np.float32)).to(device, non_blocking=True)
                    distances = distances[..., :mm]
                    quality_weights = compute_quality_weights(distances, alpha=params.get('alpha', 2.0))

            noisy_img = img_bank[0].unsqueeze(0).permute(0, 3, 1, 2)  # 1,C,H,W

            scheduler = MultiStepLR(optimizer,
                                    milestones=[int(params['epochs_per_iter'] * 0.5),
                                                int(params['epochs_per_iter'] * 0.67),
                                                int(params['epochs_per_iter'] * 0.83)],
                                    gamma=0.5)

            # Training loop
            for epoch in range(int(params['epochs_per_iter'])):
                loss_val = train_step(model, optimizer, img_bank, quality_weights, params)
                scheduler.step()

            # Denoise (for next bank rebuild or final output)
            with torch.no_grad():
                with torch.autocast(device_type=device_type_for_autocast, enabled=GPU_CONFIG['use_amp'] and device.type == 'cuda'):
                    denoised_img = torch.clamp(model(noisy_img), 0, 1)

            # Rebuild bank using denoised image (except last iteration)
            if iteration < num_iters - 1:
                topk, distances = construct_pixel_bank_from_image(denoised_img, file_name_without_ext, bank_dir, params)

        # Final evaluation
        PSNR, out_img = test(model, noisy_img, clean_img_tensor)

        # Calculate SSIM (convert pred to uint8 H,W,3)
        out_img_cpu = out_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_img_np = (np.clip(out_img_cpu, 0, 1) * 255.0).astype(np.uint8)

        # fix win_size for SSIM
        min_dim = min(clean_img_np.shape[0], clean_img_np.shape[1])
        win_size = min(7, min_dim if (min_dim % 2 == 1) else (min_dim - 1))
        try:
            SSIM, _ = compare_ssim(clean_img_np, out_img_np, full=True, channel_axis=2, win_size=max(3, win_size))
        except Exception:
            # fall back if win_size invalid or small images
            SSIM = compare_ssim(clean_img_np, out_img_np, channel_axis=2)

        if verbose:
            print(f"Image: {image_file} | PSNR: {PSNR:.2f} dB | SSIM: {SSIM:.4f}")

        # Save denoised image
        out_pil = to_pil_image(torch.from_numpy(out_img_np).permute(2, 0, 1))
        out_save_path = os.path.join(args.out_image, f"{file_name_without_ext}_denoised.png")
        out_pil.save(out_save_path)

        avg_PSNR += PSNR
        avg_SSIM += SSIM
        num_images += 1

        # Clear memory
        del model, optimizer, img_bank, quality_weights, noisy_img, denoised_img
        torch.cuda.empty_cache()

    avg_PSNR /= num_images if num_images > 0 else 1.0
    avg_SSIM /= num_images if num_images > 0 else 1.0

    return avg_PSNR, avg_SSIM

# ========================================================================
# OPTUNA OBJECTIVE FUNCTION
# ========================================================================

def objective(trial, noise_type, noise_level):
    params = {
        'layer_schedule': trial.suggest_categorical('layer_schedule', OPTUNA_CONFIG['search_space']['layer_schedule']),
        'num_iterations': trial.suggest_categorical('num_iterations', OPTUNA_CONFIG['search_space']['num_iterations']),
        'epochs_per_iter': trial.suggest_categorical('epochs_per_iter', OPTUNA_CONFIG['search_space']['epochs_per_iter']),
        'chan_embed': trial.suggest_categorical('chan_embed', OPTUNA_CONFIG['search_space']['chan_embed']),
    }

    params.update(FIXED_PARAMS)
    params['nt'] = noise_type
    params['nl'] = noise_level

    print(f"\n{'='*60}")
    print(f"Trial {trial.number} | {noise_type}-{noise_level}")
    print(f"Params: {params}")
    print(f"{'='*60}\n")

    try:
        avg_psnr, avg_ssim = denoise_images(params, verbose=False)

        trial.set_user_attr('avg_ssim', avg_ssim)
        trial.set_user_attr('noise_type', noise_type)
        trial.set_user_attr('noise_level', noise_level)

        print(f"Trial {trial.number} complete: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")

        return avg_psnr

    except Exception as e:
        print(f"Trial {trial.number} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        # prune this trial to avoid crashing whole study
        raise optuna.TrialPruned()

# ========================================================================
# RUN OPTUNA
# ========================================================================

def run_optuna():
    results = {}

    for noise_type, noise_level in OPTUNA_CONFIG['noise_configs']:
        print(f"\n{'#'*60}")
        print(f"# OPTIMIZING FOR: {noise_type.upper()} - LEVEL {noise_level}")
        print(f"{'#'*60}\n")

        study_name = f"{args.study_name}_{noise_type}_{noise_level}"

        study = optuna.create_study(
            study_name=study_name,
            storage=args.storage,
            direction='maximize',
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=123)
        )

        study.optimize(
            lambda trial: objective(trial, noise_type, noise_level),
            n_trials=OPTUNA_CONFIG['n_trials'],
            timeout=OPTUNA_CONFIG['timeout'],
            n_jobs=OPTUNA_CONFIG['n_jobs'],
            show_progress_bar=True
        )

        print(f"\n{'='*60}")
        print(f"RESULTS FOR {noise_type.upper()}-{noise_level}")
        print(f"{'='*60}")
        if study.best_value is not None:
            print(f"Best PSNR: {study.best_value:.2f} dB")
        print(f"Best params: {study.best_params}")
        best_ssim = study.best_trial.user_attrs.get('avg_ssim', 0) if study.best_trial else 0
        print(f"Best SSIM: {best_ssim:.4f}")

        results[f"{noise_type}_{noise_level}"] = {
            'best_psnr': study.best_value,
            'best_ssim': best_ssim,
            'best_params': study.best_params,
            'n_trials': len(study.trials)
        }

        result_file = f'optuna_results_{noise_type}_{noise_level}.json'
        with open(result_file, 'w') as f:
            json.dump({
                'best_psnr': study.best_value,
                'best_ssim': best_ssim,
                'best_params': study.best_params,
                'all_trials': len(study.trials),
                'completed_trials': len([t for t in study.trials if t.state == TrialState.COMPLETE])
            }, f, indent=2)

        print(f"Results saved to {result_file}\n")

    # Print summary
    print(f"\n{'#'*60}")
    print(f"# OPTIMIZATION SUMMARY")
    print(f"{'#'*60}\n")
    for config_name, result in results.items():
        print(f"{config_name:20s} | PSNR: {result['best_psnr']:6.2f} | SSIM: {result['best_ssim']:.4f}")

    with open('optuna_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nOverall summary saved to optuna_summary.json")

# ========================================================================
# SINGLE RUN MODE
# ========================================================================

def run_single():
    params = {
        'ws': args.ws,
        'ps': args.ps,
        'nn': args.nn,
        'mm': args.mm,
        'nl': args.nl,
        'nt': args.nt,
        'loss': args.loss,
        'num_iterations': args.num_iterations,
        'epochs_per_iter': args.epochs_per_iter,
        'use_quality_weights': args.use_quality_weights,
        'alpha': args.alpha,
        'progressive_growing': args.progressive_growing,
        'layer_schedule': args.layer_schedule,
        'lr': args.lr,
        'chan_embed': args.chan_embed,
    }

    print(f"\n{'='*60}")
    print("SINGLE RUN MODE")
    print(f"{'='*60}")
    print(f"Configuration:")
    for key, value in params.items():
        print(f"  {key:20s}: {value}")
    print(f"{'='*60}\n")

    avg_psnr, avg_ssim = denoise_images(params, verbose=True)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"{'='*60}\n")

# ========================================================================
# MAIN
# ========================================================================

if __name__ == "__main__":
    print("="*60)
    print("Pixel2Pixel - GPU Optimized with Optuna")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"AMP Enabled: {GPU_CONFIG['use_amp']}")
    print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
    print("="*60)

    if args.mode == 'optuna':
        print("\nStarting Optuna hyperparameter optimization...")
        print(f"Will test {len(OPTUNA_CONFIG['noise_configs'])} noise configurations")
        print(f"Trials per configuration: {OPTUNA_CONFIG['n_trials']}")
        print(f"Total trials: {len(OPTUNA_CONFIG['noise_configs']) * OPTUNA_CONFIG['n_trials']}")
        run_optuna()
    else:
        print("\nStarting single run...")
        run_single()

    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)
