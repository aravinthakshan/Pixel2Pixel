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
parser = argparse.ArgumentParser('Pixel2Pixel')
parser.add_argument('--data_path', default='./data', type=str, help='Path to the data')
parser.add_argument('--dataset', default='kodak', type=str, help='Dataset name')
parser.add_argument('--save', default='./results', type=str, help='Directory to save pixel bank results')
parser.add_argument('--out_image', default='./results_image', type=str, help='Directory to save denoised images')
parser.add_argument('--ws', default=40, type=int, help='Window size')
parser.add_argument('--ps', default=7, type=int, help='Patch size')
parser.add_argument('--nn', default=16, type=int, help='Number of nearest neighbors to search')
parser.add_argument('--mm', default=8, type=int, help='Number of pixels in pixel bank to use for training')
parser.add_argument('--nl', default=0.2, type=float, help='Noise level, for saltpepper and impulse noise, enter half the noise level.')
parser.add_argument('--nt', default='bernoulli', type=str, help='Noise type: gauss, poiss, saltpepper, bernoulli, impulse')
parser.add_argument('--loss', default='L1', type=str, help='Loss function type')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of bank reconstruction iterations')
parser.add_argument('--epochs_per_iter', default=1000, type=int, help='Epochs per iteration')
parser.add_argument('--use_quality_weights', default=True, type=bool, help='Use quality-based sampling weights')
parser.add_argument('--alpha', default=2.0, type=float, help='Sharpness of quality scoring (higher = more selective)')
parser.add_argument('--progressive_growing', default=False, type=bool, help='Use progressive network growing')
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
    if noise_type == 'gauss':
        noisy = x + torch.normal(0, noise_level / 255, x.shape)
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


# This is the only change Working Condition
# -------------------------------
def construct_pixel_bank_from_image(img_tensor, file_name_without_ext, bank_dir):
    """
    Construct pixel bank from a given image tensor.
    img_tensor: [1, C, H, W] tensor on GPU
    Returns: topk tensor and distance tensor for quality scoring
    """
    pad_sz = WINDOW_SIZE // 2 + PATCH_SIZE // 2
    center_offset = WINDOW_SIZE // 2
    blk_sz = 64  # Block size for processing

    img = img_tensor  # Already on GPU

    # Pad image and extract patches
    img_pad = F.pad(img, (pad_sz, pad_sz, pad_sz, pad_sz), mode='reflect')
    img_unfold = F.unfold(img_pad, kernel_size=PATCH_SIZE, padding=0, stride=1)
    H_new = img.shape[-2] + WINDOW_SIZE
    W_new = img.shape[-1] + WINDOW_SIZE
    img_unfold = einops.rearrange(img_unfold, 'b c (h w) -> b c h w', h=H_new, w=W_new)

    num_blk_w = img.shape[-1] // blk_sz
    num_blk_h = img.shape[-2] // blk_sz
    is_window_size_even = (WINDOW_SIZE % 2 == 0)
    topk_list = []
    distance_list = []

    # Iterate over blocks in the image
    for blk_i in range(num_blk_w):
        for blk_j in range(num_blk_h):
            start_h = blk_j * blk_sz
            end_h = (blk_j + 1) * blk_sz + WINDOW_SIZE
            start_w = blk_i * blk_sz
            end_w = (blk_i + 1) * blk_sz + WINDOW_SIZE

            sub_img_uf = img_unfold[..., start_h:end_h, start_w:end_w]
            sub_img_shape = sub_img_uf.shape

            if is_window_size_even:
                sub_img_uf_inp = sub_img_uf[..., :-1, :-1]
            else:
                sub_img_uf_inp = sub_img_uf

            patch_windows = F.unfold(sub_img_uf_inp, kernel_size=WINDOW_SIZE, padding=0, stride=1)
            patch_windows = einops.rearrange(
                patch_windows,
                'b (c k1 k2 k3 k4) (h w) -> b (c k1 k2) (k3 k4) h w',
                k1=PATCH_SIZE, k2=PATCH_SIZE, k3=WINDOW_SIZE, k4=WINDOW_SIZE,
                h=blk_sz, w=blk_sz
            )

            img_center = einops.rearrange(
                sub_img_uf,
                'b (c k1 k2) h w -> b (c k1 k2) 1 h w',
                k1=PATCH_SIZE, k2=PATCH_SIZE,
                h=sub_img_shape[-2], w=sub_img_shape[-1]
            )
            img_center = img_center[..., center_offset:center_offset + blk_sz, center_offset:center_offset + blk_sz]

            if args.loss == 'L2':
                distance = torch.sum((img_center - patch_windows) ** 2, dim=1)
            elif args.loss == 'L1':
                distance = torch.sum(torch.abs(img_center - patch_windows), dim=1)
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")

            topk_distances, sort_indices = torch.topk(
                distance,
                k=NUM_NEIGHBORS,
                largest=False,
                sorted=True,
                dim=-3
            )

            patch_windows_reshape = einops.rearrange(
                patch_windows,
                'b (c k1 k2) (k3 k4) h w -> b c (k1 k2) (k3 k4) h w',
                k1=PATCH_SIZE, k2=PATCH_SIZE, k3=WINDOW_SIZE, k4=WINDOW_SIZE
            )
            patch_center = patch_windows_reshape[:, :, patch_windows_reshape.shape[2] // 2, ...]
            topk = torch.gather(patch_center, dim=-3,
                                index=sort_indices.unsqueeze(1).repeat(1, 3, 1, 1, 1))
            topk_list.append(topk)
            distance_list.append(topk_distances)

    # Merge the results from all blocks to form the pixel bank
    topk = torch.cat(topk_list, dim=0)
    topk = einops.rearrange(topk, '(w1 w2) c k h w -> k c (w2 h) (w1 w)', w1=num_blk_w, w2=num_blk_h)
    topk = topk.permute(2, 3, 0, 1)
    
    distances = torch.cat(distance_list, dim=0)
    distances = einops.rearrange(distances, '(w1 w2) k h w -> k (w2 h) (w1 w)', w1=num_blk_w, w2=num_blk_h)
    distances = distances.permute(1, 2, 0)

    # Save pixel bank and distances
    np.save(os.path.join(bank_dir, file_name_without_ext), topk.cpu().numpy())
    np.save(os.path.join(bank_dir, file_name_without_ext + '_distances'), distances.cpu().numpy())
    
    return topk, distances


def construct_pixel_bank():
    bank_dir = os.path.join(args.save, '_'.join(
        str(i) for i in [args.dataset, args.nt, args.nl, args.ws, args.ps, args.nn, args.loss]))
    os.makedirs(bank_dir, exist_ok=True)

    image_folder = os.path.join(args.data_path, args.dataset)
    image_files = sorted(os.listdir(image_folder))

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        start_time = time.time()

        # Load image and add noise
        img = Image.open(image_path)
        img = transform(img).unsqueeze(0)  # Shape: [1, C, H, W]
        img = add_noise(img, noise_level).squeeze(0)
        img = img.cuda()[None, ...]  # Shape: [1, C, H, W]

        file_name_without_ext = os.path.splitext(image_file)[0]
        topk, distances = construct_pixel_bank_from_image(img, file_name_without_ext, bank_dir)

        elapsed = time.time() - start_time
        print(f"Processed {image_file} in {elapsed:.2f} seconds. Pixel bank shape: {topk.shape}")

    print("Pixel bank construction completed for all images.")

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


def mse_loss(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(gt, pred)


loss_f = nn.L1Loss() if args.loss == 'L1' else nn.MSELoss()


def loss_func(img1, img2, loss_f=nn.MSELoss()):
    pred1 = model(img1)
    loss = loss_f(img2, pred1)
    return loss



# # -------------------------------
# def compute_quality_weights(distances, alpha=2.0):
#     """
#     Compute quality weights for pixel bank samples based on distances.
    
#     Args:
#         distances: [H, W, K] tensor of distances for each pixel's K neighbors
#         alpha: Sharpness parameter (higher = more selective)
    
#     Returns:
#         weights: [H, W, K] tensor of sampling weights
#     """
#     # Normalize distances per pixel to [0, 1]
#     # Shape: [H, W, K]
#     dist_min = distances.min(dim=-1, keepdim=True)[0]
#     dist_max = distances.max(dim=-1, keepdim=True)[0]
#     dist_range = dist_max - dist_min
#     dist_range = torch.clamp(dist_range, min=1e-8)  # Avoid division by zero
#     normalized_dist = (distances - dist_min) / dist_range
    
#     # Quality score: Gaussian-like curve peaked at medium distances
#     # We want to favor patches with normalized distance around 0.3-0.7
#     optimal_distance = 0.5
#     quality_scores = torch.exp(-alpha * (normalized_dist - optimal_distance) ** 2)
    
#     # Penalize very similar patches (distance ≈ 0) more heavily
#     too_similar_penalty = torch.exp(-10 * normalized_dist)
#     quality_scores = quality_scores * (1 - 0.5 * too_similar_penalty)
    
#     # Normalize to get sampling probabilities
#     weights = quality_scores / (quality_scores.sum(dim=-1, keepdim=True) + 1e-8)
    
#     return weights

def compute_quality_weights(distances, patch_features=None, lambda_param=0.2, alpha=2.0):
    """
    Compute MMR-based quality weights for pixel bank samples.
    
    MMR balances relevance (low distance to center pixel) with diversity 
    (dissimilarity among selected patches).
    
    Args:
        distances: [H, W, K] tensor of distances for each pixel's K neighbors
        patch_features: [H, W, K, D] optional tensor of patch features for diversity computation
                       If None, uses distances to approximate diversity
        lambda_param: Trade-off between relevance and diversity (0=diversity only, 1=relevance only)
        alpha: Sharpness parameter for converting scores to weights
    
    Returns:
        weights: [H, W, K] tensor of sampling weights based on MMR scores
    """
    H, W, K = distances.shape
    device = distances.device
    
    # Normalize distances to [0, 1] for relevance scores
    dist_min = distances.min(dim=-1, keepdim=True)[0]
    dist_max = distances.max(dim=-1, keepdim=True)[0]
    dist_range = torch.clamp(dist_max - dist_min, min=1e-8)
    normalized_dist = (distances - dist_min) / dist_range
    
    # Relevance score: Lower distance = higher relevance
    relevance = 1.0 - normalized_dist  # [H, W, K]
    
    # Compute diversity scores
    if patch_features is not None:
        # Use actual patch features for diversity computation
        # Compute pairwise similarities between patches at each pixel
        # patch_features: [H, W, K, D]
        feat_norm = F.normalize(patch_features, p=2, dim=-1)  # L2 normalize
        # Compute similarity matrix: [H, W, K, K]
        similarity_matrix = torch.einsum('hwid,hwjd->hwij', feat_norm, feat_norm)
        # Diversity = average dissimilarity to other patches
        diversity = 1.0 - similarity_matrix.mean(dim=-1)  # [H, W, K]
    else:
        # Approximate diversity using distance distribution
        # Patches with unique distances are more diverse
        # Compute variance of distances as a proxy for diversity
        dist_mean = distances.mean(dim=-1, keepdim=True)
        dist_variance = ((distances - dist_mean) ** 2).mean(dim=-1, keepdim=True)
        dist_std = torch.sqrt(dist_variance + 1e-8)
        
        # Diversity score: patches with distances far from mean are more diverse
        diversity = torch.abs(distances - dist_mean) / (dist_std + 1e-8)
        diversity = torch.tanh(diversity)  # Normalize to [0, 1]
    
    # MMR score: Linear combination of relevance and diversity
    mmr_scores = lambda_param * relevance + (1 - lambda_param) * diversity
    
    # Apply sharpening to create more selective weights
    # Use softmax with temperature for smooth probability distribution
    temperature = 1.0 / alpha
    mmr_scores_scaled = mmr_scores / temperature
    
    # Convert to probabilities using softmax along K dimension
    weights = F.softmax(mmr_scores_scaled, dim=-1)
    
    # Optional: Apply additional shaping to avoid very uniform distributions
    # Boost high-scoring samples
    weights = weights ** (1.0 + alpha * 0.1)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
    
    return weights

def train(model, optimizer, img_bank, quality_weights=None):
    N, H, W, C = img_bank.shape
    
    if args.use_quality_weights and quality_weights is not None:
        # Sample based on quality weights
        # quality_weights shape: [H, W, N]
        # Flatten for easier sampling
        flat_weights = quality_weights.view(-1, N)  # [H*W, N]
        
        # Sample indices for each pixel based on quality weights
        index1 = torch.multinomial(flat_weights, num_samples=1, replacement=True).view(H, W)
        
        # Sample second index differently (also based on weights)
        index2 = torch.multinomial(flat_weights, num_samples=1, replacement=True).view(H, W)
        
        # Ensure different indices
        eq_mask = (index2 == index1)
        if eq_mask.any():
            # Resample for equal indices
            eq_positions = eq_mask.view(-1)
            eq_weights = flat_weights[eq_positions]
            new_samples = torch.multinomial(eq_weights, num_samples=1, replacement=True)
            index2.view(-1)[eq_positions] = new_samples.squeeze()
            # If still equal after one resample, just add 1 mod N
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

    loss = loss_func(img1, img2, loss_f)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_original(model, optimizer, img_bank):
    """Original training function without quality weights"""
    N, H, W, C = img_bank.shape
    index1 = torch.randint(0, N, size=(H, W), device=device)
    index1_exp = index1.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img1 = torch.gather(img_bank, 0, index1_exp)  # Result shape: (1, H, W, C)
    img1 = img1.permute(0, 3, 1, 2)

    index2 = torch.randint(0, N, size=(H, W), device=device)
    eq_mask = (index2 == index1)
    if eq_mask.any():
        index2[eq_mask] = (index2[eq_mask] + 1) % N
    index2_exp = index2.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img2 = torch.gather(img_bank, 0, index2_exp)
    img2 = img2.permute(0, 3, 1, 2)

    loss = loss_func(img1, img2, loss_f)
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


# -------------------------------
def denoise_images():
    bank_dir = os.path.join(args.save, '_'.join(
        str(i) for i in [args.dataset, args.nt, args.nl, args.ws, args.ps, args.nn, args.loss]))
    image_folder = os.path.join(args.data_path, args.dataset)
    image_files = sorted(os.listdir(image_folder))

    os.makedirs(args.out_image, exist_ok=True)

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
        global model
        model = Network(n_chan).to(device)
        print(f"Number of parameters for {image_file}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        
        # Iterative bank reconstruction
        for iteration in range(args.num_iterations):
            print(f"\n{'='*60}")
            print(f"Image: {image_file} | Iteration {iteration + 1}/{args.num_iterations}")
            print(f"{'='*60}")
            
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
                    # Only use distances for the selected mm samples
                    distances = distances[..., :args.mm]
                    quality_weights = compute_quality_weights(distances, alpha=args.alpha) # distances, patch_features=None, lambda_param=0.5, alpha=2.0 func signature
                    print(f"Using quality-weighted sampling (alpha={args.alpha})")
                    # Print some statistics
                    avg_weight = quality_weights.mean().item()
                    max_weight = quality_weights.max().item()
                    min_weight = quality_weights.min().item()
                    print(f"  Weight stats - Mean: {avg_weight:.4f}, Max: {max_weight:.4f}, Min: {min_weight:.4f}")
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
            for epoch in range(args.epochs_per_iter):
                train(model, optimizer, img_bank, quality_weights)
                scheduler.step()
                
                if (epoch + 1) % 200 == 0:
                    with torch.no_grad():
                        current_pred = torch.clamp(model(noisy_img), 0, 1)
                        current_mse = mse_loss(clean_img_tensor, current_pred).item()
                        current_psnr = 10 * np.log10(1 / current_mse)
                    print(f"  Epoch {epoch+1}/{args.epochs_per_iter} - PSNR: {current_psnr:.2f} dB")

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
                topk, distances = construct_pixel_bank_from_image(denoised_img, file_name_without_ext, bank_dir)
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
        # Determine appropriate win_size based on image dimensions
        min_dim = min(clean_img_np.shape[0], clean_img_np.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        SSIM, _ = compare_ssim(clean_img_np, out_img_loaded, full=True, channel_axis=2, win_size=win_size)
        print(f"\nFinal Results - Image: {image_file} | PSNR: {PSNR:.2f} dB | SSIM: {SSIM:.4f}")
        avg_PSNR += PSNR
        avg_SSIM += SSIM

    avg_PSNR /= len(image_files)
    avg_SSIM /= len(image_files)
    print(f"\n{'='*60}")
    print(f"Average PSNR: {avg_PSNR:.2f} dB, Average SSIM: {avg_SSIM:.4f}")
    print(f"{'='*60}")

# -------------------------------
if __name__ == "__main__":
    print("="*60)
    print("Pixel2Pixel with Iterative Refinement & Quality Weighting")
    print("="*60)
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Noise type: {args.nt}, level: {args.nl}")
    print(f"  Window size: {args.ws}, Patch size: {args.ps}")
    print(f"  Neighbors: {args.nn}, Loss: {args.loss}")
    print(f"  Iterations: {args.num_iterations}, Epochs/iter: {args.epochs_per_iter}")
    print(f"  Quality weighting: {args.use_quality_weights}, Alpha: {args.alpha}")
    print("="*60)
    print("\nConstructing initial pixel banks from noisy images...")
    construct_pixel_bank()
    print("\nStarting iterative denoising with bank reconstruction...")
    denoise_images()