import os
import time
import argparse
import numpy as np
from PIL import Image
from skimage import io
from skimage.metrics import structural_similarity as compare_ssim
import json

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
    # Test configurations for Optuna
    'noise_configs': [
        ('gauss', 10),
        ('gauss', 25),
        ('gauss', 30),
        ('poiss', 10),
        ('poiss', 25),
        ('poiss', 50),
    ],
    
    # Hyperparameter search spaces
    'search_space': {
        'layer_schedule': ['3,5,6', '6,9,12', '4,8,12', '5,10,15'],
        'alpha': [1.5, 2.0, 2.5, 3.0],
        'curriculum_start': [0.8, 1.0],
        'curriculum_end': [0.15, 0.25, 0.35],
        'num_iterations': [2, 3, 4],
        'epochs_per_iter': [800, 1000, 1200],
        'lr': [0.0005, 0.001, 0.0015],
        'chan_embed': [48, 64, 96],
    },
    
    # Optuna settings
    'n_trials': 100,  # Number of trials to run
    'timeout': None,  # Timeout in seconds (None = no timeout)
    'n_jobs': 1,  # Parallel jobs (1 for GPU to avoid OOM)
}

# GPU Optimization settings
GPU_CONFIG = {
    'use_amp': True,  # Automatic Mixed Precision for A100
    'batch_accumulation': 2,  # Gradient accumulation steps
    'pin_memory': True,  # Faster CPU-GPU transfer
    'num_workers': 4,  # For data loading (if applicable)
    'prefetch_factor': 2,  # Prefetch batches
}

# Fixed parameters
FIXED_PARAMS = {
    'ws': 40,
    'ps': 7,
    'nn': 16,
    'mm': 8,  # Will be adjusted based on noise
    'loss': 'L1',
    'use_quality_weights': True,
    'progressive_growing': True,
    'curriculum_learning': True,
}

# ========================================================================
# ARGUMENT PARSER
# ========================================================================

parser = argparse.ArgumentParser('Pixel2Pixel with Optuna Optimization')
parser.add_argument('--data_path', default='./data', type=str)
parser.add_argument('--dataset', default='kodak', type=str)
parser.add_argument('--save', default='./results', type=str)
parser.add_argument('--out_image', default='./results_image', type=str)
parser.add_argument('--mode', default='optuna', type=str, choices=['optuna', 'single'], 
                    help='Run mode: optuna for hyperparameter search, single for one run')
parser.add_argument('--study_name', default='pixel2pixel_study', type=str)
parser.add_argument('--storage', default='sqlite:///optuna_study.db', type=str)

# Override parameters for single mode
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
parser.add_argument('--curriculum_learning', default=True, type=bool)
parser.add_argument('--curriculum_start', default=1.0, type=float)
parser.add_argument('--curriculum_end', default=0.25, type=float)
parser.add_argument('--layer_schedule', default='3,5,6', type=str)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--chan_embed', default=64, type=int)

args = parser.parse_args()

# ========================================================================
# SETUP
# ========================================================================

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # Enable for A100 optimization
torch.backends.cuda.matmul.allow_tf32 = True  # A100 tensor cores
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

transform = transforms.Compose([transforms.ToTensor()])

# Global variables for current run
current_noise_type = None
current_noise_level = None
current_params = None

# ========================================================================
# NOISE FUNCTIONS
# ========================================================================

def add_noise(x, noise_level, noise_type):
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

# ========================================================================
# PIXEL BANK CONSTRUCTION (GPU-OPTIMIZED)
# ========================================================================

@torch.cuda.amp.autocast(enabled=GPU_CONFIG['use_amp'])
def construct_pixel_bank_from_image(img_tensor, file_name_without_ext, bank_dir, params):
    pad_sz = params['ws'] // 2 + params['ps'] // 2
    center_offset = params['ws'] // 2
    blk_sz = 128  # Larger blocks for A100

    img = img_tensor

    # Pad and unfold - keep on GPU
    img_pad = F.pad(img, (pad_sz, pad_sz, pad_sz, pad_sz), mode='reflect')
    img_unfold = F.unfold(img_pad, kernel_size=params['ps'], padding=0, stride=1)
    H_new = img.shape[-2] + params['ws']
    W_new = img.shape[-1] + params['ws']
    img_unfold = einops.rearrange(img_unfold, 'b c (h w) -> b c h w', h=H_new, w=W_new)

    num_blk_w = img.shape[-1] // blk_sz
    num_blk_h = img.shape[-2] // blk_sz
    is_window_size_even = (params['ws'] % 2 == 0)
    
    topk_list = []
    distance_list = []

    # Process blocks with GPU optimization
    with torch.cuda.amp.autocast(enabled=GPU_CONFIG['use_amp']):
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
                elif params['loss'] == 'L1':
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
                topk = torch.gather(patch_center, dim=-3,
                                    index=sort_indices.unsqueeze(1).repeat(1, 3, 1, 1, 1))
                topk_list.append(topk)
                distance_list.append(topk_distances)

    # Merge results
    topk = torch.cat(topk_list, dim=0)
    topk = einops.rearrange(topk, '(w1 w2) c k h w -> k c (w2 h) (w1 w)', w1=num_blk_w, w2=num_blk_h)
    topk = topk.permute(2, 3, 0, 1)
    
    distances = torch.cat(distance_list, dim=0)
    distances = einops.rearrange(distances, '(w1 w2) k h w -> k (w2 h) (w1 w)', w1=num_blk_w, w2=num_blk_h)
    distances = distances.permute(1, 2, 0)

    # Save to disk (move to CPU for saving)
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

        img = Image.open(image_path)
        img = transform(img).unsqueeze(0)
        img = add_noise(img, params['nl'], params['nt']).squeeze(0)
        img = img.to(device)[None, ...]

        file_name_without_ext = os.path.splitext(image_file)[0]
        topk, distances = construct_pixel_bank_from_image(img, file_name_without_ext, bank_dir, params)

        elapsed = time.time() - start_time
        print(f"Processed {image_file} in {elapsed:.2f}s. Bank shape: {topk.shape}")

    print("Pixel bank construction completed.")
    return bank_dir

# ========================================================================
# NETWORK (GPU-OPTIMIZED)
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
                    init.constant_(m.bias, 0)

# ========================================================================
# QUALITY WEIGHTS & TRAINING
# ========================================================================

def compute_quality_weights(distances, alpha=2.0, curriculum_threshold=1.0):
    dist_min = distances.min(dim=-1, keepdim=True)[0]
    dist_max = distances.max(dim=-1, keepdim=True)[0]
    dist_range = torch.clamp(dist_max - dist_min, min=1e-8)
    normalized_dist = (distances - dist_min) / dist_range
    
    optimal_distance = 0.5
    quality_scores = torch.exp(-alpha * (normalized_dist - optimal_distance) ** 2)
    too_similar_penalty = torch.exp(-10 * normalized_dist)
    quality_scores = quality_scores * (1 - 0.5 * too_similar_penalty)
    
    if curriculum_threshold < 1.0:
        K = quality_scores.shape[-1]
        k_to_keep = max(1, int(K * curriculum_threshold))
        threshold_values, _ = torch.kthvalue(quality_scores, K - k_to_keep + 1, dim=-1, keepdim=True)
        curriculum_mask = (quality_scores >= threshold_values).float()
        quality_scores = quality_scores * curriculum_mask
    
    weights = quality_scores / (quality_scores.sum(dim=-1, keepdim=True) + 1e-8)
    return weights


def train_step(model, optimizer, img_bank, quality_weights, scaler, params):
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
    img1 = torch.gather(img_bank, 0, index1_exp).permute(0, 3, 1, 2)
    
    index2_exp = index2.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img2 = torch.gather(img_bank, 0, index2_exp).permute(0, 3, 1, 2)
    
    loss_f = nn.L1Loss() if params['loss'] == 'L1' else nn.MSELoss()
    
    with torch.cuda.amp.autocast(enabled=GPU_CONFIG['use_amp']):
        pred = model(img1)
        loss = loss_f(img2, pred)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
    
    return loss.item()


def test(model, noisy_img, clean_img):
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=GPU_CONFIG['use_amp']):
            pred = torch.clamp(model(noisy_img), 0, 1)
        mse_val = F.mse_loss(clean_img, pred).item()
        psnr = 10 * np.log10(1 / mse_val) if mse_val > 0 else 100.0
    return psnr, pred

# ========================================================================
# MAIN DENOISING FUNCTION
# ========================================================================

def denoise_images(params, verbose=True):
    bank_dir = os.path.join(args.save, '_'.join(
        str(i) for i in [args.dataset, params['nt'], params['nl'], params['ws'], params['ps'], params['nn'], params['loss']]))
    
    # Check if bank exists, create if not
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

    avg_PSNR = 0
    avg_SSIM = 0
    num_images = 0

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        clean_img = Image.open(image_path)
        clean_img_tensor = transform(clean_img).unsqueeze(0).to(device)
        clean_img_np = io.imread(image_path)

        file_name_without_ext = os.path.splitext(image_file)[0]
        bank_path = os.path.join(bank_dir, file_name_without_ext)
        
        if not os.path.exists(bank_path + '.npy'):
            continue

        # Adjust mm based on noise
        if params['nt'] == 'gauss' and params['nl'] == 10 or params['nt'] == 'bernoulli':
            mm = 2
        elif params['nt'] == 'gauss' and params['nl'] == 25:
            mm = 4
        else:
            mm = 8

        n_chan = clean_img_tensor.shape[1]
        
        # Parse layer schedule
        layer_schedule_config = [int(x.strip()) for x in params['layer_schedule'].split(',')]
        max_layers_needed = max(layer_schedule_config)
        
        model = Network(n_chan, chan_embed=params['chan_embed'], max_layers=max_layers_needed).to(device)
        
        # Generate layer schedule
        if len(layer_schedule_config) == params['num_iterations']:
            layer_schedule = layer_schedule_config
        elif len(layer_schedule_config) == 1:
            base_layers = layer_schedule_config[0]
            if params['progressive_growing']:
                if params['num_iterations'] == 1:
                    layer_schedule = [base_layers]
                else:
                    min_layers = max(2, base_layers - params['num_iterations'] + 1)
                    layer_schedule = []
                    for i in range(params['num_iterations']):
                        progress = i / (params['num_iterations'] - 1)
                        layers = int(min_layers + progress * (base_layers - min_layers))
                        layer_schedule.append(layers)
            else:
                layer_schedule = [base_layers] * params['num_iterations']
        else:
            layer_schedule = layer_schedule_config[:params['num_iterations']]
        
        scaler = torch.cuda.amp.GradScaler(enabled=GPU_CONFIG['use_amp'])

        # Iterative training
        for iteration in range(params['num_iterations']):
            current_layers = layer_schedule[iteration]
            model.set_active_layers(current_layers)
            
            if params['progressive_growing']:
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
            if params['use_quality_weights']:
                dist_path = os.path.join(bank_dir, file_name_without_ext + '_distances.npy')
                if os.path.exists(dist_path):
                    distances_arr = np.load(dist_path)
                    distances = torch.from_numpy(distances_arr.astype(np.float32)).to(device, non_blocking=True)
                    distances = distances[..., :mm]
                    
                    if params['curriculum_learning']:
                        progress = iteration / max(1, params['num_iterations'] - 1)
                        curriculum_threshold = params['curriculum_start'] - progress * (params['curriculum_start'] - params['curriculum_end'])
                    else:
                        curriculum_threshold = 1.0
                    
                    quality_weights = compute_quality_weights(distances, alpha=params['alpha'], curriculum_threshold=curriculum_threshold)

            noisy_img = img_bank[0].unsqueeze(0).permute(0, 3, 1, 2)
            
            scheduler = MultiStepLR(optimizer, 
                                   milestones=[int(params['epochs_per_iter']*0.5), 
                                             int(params['epochs_per_iter']*0.67), 
                                             int(params['epochs_per_iter']*0.83)], 
                                   gamma=0.5)

            # Training loop
            for epoch in range(params['epochs_per_iter']):
                train_step(model, optimizer, img_bank, quality_weights, scaler, params)
                scheduler.step()

            # Denoise
            with torch.no_grad():
                denoised_img = torch.clamp(model(noisy_img), 0, 1)
            
            # Rebuild bank (except last iteration)
            if iteration < params['num_iterations'] - 1:
                topk, distances = construct_pixel_bank_from_image(denoised_img, file_name_without_ext, bank_dir, params)

        # Final evaluation
        PSNR, out_img = test(model, noisy_img, clean_img_tensor)
        
        # Calculate SSIM
        out_img_np = (out_img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        min_dim = min(clean_img_np.shape[0], clean_img_np.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        SSIM, _ = compare_ssim(clean_img_np, out_img_np, full=True, channel_axis=2, win_size=win_size)
        
        if verbose:
            print(f"Image: {image_file} | PSNR: {PSNR:.2f} dB | SSIM: {SSIM:.4f}")
        
        avg_PSNR += PSNR
        avg_SSIM += SSIM
        num_images += 1
        
        # Clear memory
        del model, optimizer, img_bank, quality_weights
        torch.cuda.empty_cache()

    avg_PSNR /= num_images if num_images > 0 else 1
    avg_SSIM /= num_images if num_images > 0 else 1
    
    return avg_PSNR, avg_SSIM

# ========================================================================
# OPTUNA OBJECTIVE FUNCTION
# ========================================================================

def objective(trial, noise_type, noise_level):
    # Sample hyperparameters
    params = {
        'layer_schedule': trial.suggest_categorical('layer_schedule', OPTUNA_CONFIG['search_space']['layer_schedule']),
        'alpha': trial.suggest_categorical('alpha', OPTUNA_CONFIG['search_space']['alpha']),
        'curriculum_start': trial.suggest_categorical('curriculum_start', OPTUNA_CONFIG['search_space']['curriculum_start']),
        'curriculum_end': trial.suggest_categorical('curriculum_end', OPTUNA_CONFIG['search_space']['curriculum_end']),
        'num_iterations': trial.suggest_categorical('num_iterations', OPTUNA_CONFIG['search_space']['num_iterations']),
        'epochs_per_iter': trial.suggest_categorical('epochs_per_iter', OPTUNA_CONFIG['search_space']['epochs_per_iter']),
        'lr': trial.suggest_categorical('lr', OPTUNA_CONFIG['search_space']['lr']),
        'chan_embed': trial.suggest_categorical('chan_embed', OPTUNA_CONFIG['search_space']['chan_embed']),
    }
    
    # Add fixed parameters
    params.update(FIXED_PARAMS)
    params['nt'] = noise_type
    params['nl'] = noise_level
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number} | {noise_type}-{noise_level}")
    print(f"Params: {params}")
    print(f"{'='*60}\n")
    
    try:
        avg_psnr, avg_ssim = denoise_images(params, verbose=False)
        
        # Store additional metrics
        trial.set_user_attr('avg_ssim', avg_ssim)
        trial.set_user_attr('noise_type', noise_type)
        trial.set_user_attr('noise_level', noise_level)
        
        print(f"Trial {trial.number} complete: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")
        
        return avg_psnr  # Optuna maximizes this
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {str(e)}")
        raise optuna.TrialPruned()

# ========================================================================
# RUN OPTUNA OPTIMIZATION
# ========================================================================

def run_optuna():
    results = {}
    
    for noise_type, noise_level in OPTUNA_CONFIG['noise_configs']:
        print(f"\n{'#'*60}")
        print(f"# OPTIMIZING FOR: {noise_type.upper()} - LEVEL {noise_level}")
        print(f"{'#'*60}\n")
        
        study_name = f"{args.study_name}_{noise_type}_{noise_level}"
        
        # Create or load study
        study = optuna.create_study(
            study_name=study_name,
            storage=args.storage,
            direction='maximize',
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=123)
        )
        
        # Run optimization
        study.optimize(
            lambda trial: objective(trial, noise_type, noise_level),
            n_trials=OPTUNA_CONFIG['n_trials'],
            timeout=OPTUNA_CONFIG['timeout'],
            n_jobs=OPTUNA_CONFIG['n_jobs'],
            show_progress_bar=True
        )
        
        # Print results
        print(f"\n{'='*60}")
        print(f"RESULTS FOR {noise_type.upper()}-{noise_level}")
        print(f"{'='*60}")
        print(f"Best PSNR: {study.best_value:.2f} dB")
        print(f"Best params: {study.best_params}")
        print(f"Best SSIM: {study.best_trial.user_attrs.get('avg_ssim', 'N/A'):.4f}")
        print(f"Number of trials: {len(study.trials)}")
        print(f"Number of completed trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
        
        # Store results
        results[f"{noise_type}_{noise_level}"] = {
            'best_psnr': study.best_value,
            'best_ssim': study.best_trial.user_attrs.get('avg_ssim', 0),
            'best_params': study.best_params,
            'n_trials': len(study.trials)
        }
        
        # Save results to file
        result_file = f'optuna_results_{noise_type}_{noise_level}.json'
        with open(result_file, 'w') as f:
            json.dump({
                'best_psnr': study.best_value,
                'best_ssim': study.best_trial.user_attrs.get('avg_ssim', 0),
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
    
    # Save overall summary
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
        'curriculum_learning': args.curriculum_learning,
        'curriculum_start': args.curriculum_start,
        'curriculum_end': args.curriculum_end,
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
# MAIN ENTRY POINT
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