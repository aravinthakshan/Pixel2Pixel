import os 
import einops
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from src.utils import add_noise

transform = transforms.Compose([transforms.ToTensor()])

def construct_pixel_bank_real(args):
    # The pixel banks will be saved in a directory constructed from dataset parameters
    bank_dir = os.path.join(args.bank_dir, args.dataset, '_'.join(str(i) for i in [args.ws, args.ps, args.nn, args.loss]))
    os.makedirs(bank_dir, exist_ok=True)

    WINDOW_SIZE = args.ws
    PATCH_SIZE = args.ps
    NUM_NEIGHBORS = args.nn
    loss_type = args.loss

    noisy_folder = os.path.join(args.data_path, args.dataset, args.noisy_dir)
    image_files = sorted(os.listdir(noisy_folder))

    pad_sz = WINDOW_SIZE // 2 + PATCH_SIZE // 2
    center_offset = WINDOW_SIZE // 2
    blk_sz = 64  # Block size for processing

    for image_file in image_files:
        image_path = os.path.join(noisy_folder, image_file)
        start_time = time.time()

        # Load the already noisy image
        img = Image.open(image_path)
        img = transform(img).unsqueeze(0)  # Shape: [1, C, H, W]
        img = img.cuda()  # No extra dimension is added

        # Pad the image (F.pad requires a 4D tensor)
        img_pad = F.pad(img, (pad_sz, pad_sz, pad_sz, pad_sz), mode='reflect')
        # Extract patches by unfolding the image into sliding window patches
        img_unfold = F.unfold(img_pad, kernel_size=PATCH_SIZE, padding=0, stride=1)
        H_new = img.shape[-2] + WINDOW_SIZE
        W_new = img.shape[-1] + WINDOW_SIZE
        img_unfold = einops.rearrange(img_unfold, 'b c (h w) -> b c h w', h=H_new, w=W_new)
        print(f"Image {image_file} - shape after unfolding: {img_unfold.shape}")

        num_blk_w = img.shape[-1] // blk_sz
        num_blk_h = img.shape[-2] // blk_sz
        is_window_size_even = (WINDOW_SIZE % 2 == 0)
        topk_list = []

        # Process each block
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

                # Compute L2 distances and select the most similar patches
                l2_dis = torch.sum((img_center - patch_windows) ** 2, dim=1)
                _, sort_indices = torch.topk(l2_dis, k=NUM_NEIGHBORS, largest=False, sorted=True, dim=-3)

                patch_windows_reshape = einops.rearrange(
                    patch_windows,
                    'b (c k1 k2) (k3 k4) h w -> b c (k1 k2) (k3 k4) h w',
                    k1=PATCH_SIZE, k2=PATCH_SIZE, k3=WINDOW_SIZE, k4=WINDOW_SIZE
                )
                patch_center = patch_windows_reshape[:, :, patch_windows_reshape.shape[2] // 2, ...]
                topk = torch.gather(patch_center, dim=-3,
                                    index=sort_indices.unsqueeze(1).repeat(1, 3, 1, 1, 1))
                topk_list.append(topk)

        # Merge results from all blocks to form the pixel bank
        topk = torch.cat(topk_list, dim=0)
        topk = einops.rearrange(topk, '(w1 w2) c k h w -> k c (w2 h) (w1 w)', w1=num_blk_w, w2=num_blk_h)
        topk = topk.permute(2, 3, 0, 1)

        elapsed = time.time() - start_time
        print(f"Processed {image_file} in {elapsed:.2f} seconds. Pixel bank shape: {topk.shape}")

        file_name_without_ext = os.path.splitext(image_file)[0]
        np.save(os.path.join(bank_dir, file_name_without_ext), topk.cpu())

    print("Pixel bank construction completed for all images.")


def construct_pixel_bank_from_image(img_tensor, file_name_without_ext, bank_dir, args):
    """
    Construct pixel bank from a given image tensor.
    img_tensor: [1, C, H, W] tensor on GPU
    Returns: topk tensor and distance tensor for quality scoring
    """

    WINDOW_SIZE = args.ws
    PATCH_SIZE = args.ps
    NUM_NEIGHBORS = args.nn
    loss_type = args.loss
    
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

            if loss_type == 'L2':
                distance = torch.sum((img_center - patch_windows) ** 2, dim=1)
            elif loss_type == 'L1':
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

def construct_pixel_bank(args):
    noise_level = args.nl
    noise_type = args.nt

    bank_dir = os.path.join(args.bank_dir, '_'.join(
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
        img = add_noise(img, noise_type, noise_level).squeeze(0)
        img = img.cuda()[None, ...]  # Shape: [1, C, H, W]

        file_name_without_ext = os.path.splitext(image_file)[0]
        topk, distances = construct_pixel_bank_from_image(img, file_name_without_ext, bank_dir, args)

        elapsed = time.time() - start_time
        print(f"Processed {image_file} in {elapsed:.2f} seconds. Pixel bank shape: {topk.shape}")

    print("Pixel bank construction completed for all images.")

def compute_quality_weights_distance(distances, alpha=2.0):

    # Normalize distances per pixel to [0, 1]
    dist_min = distances.min(dim=-1, keepdim=True)[0]
    dist_max = distances.max(dim=-1, keepdim=True)[0]
    dist_range = dist_max - dist_min
    dist_range = torch.clamp(dist_range, min=1e-8)
    normalized_dist = (distances - dist_min) / dist_range
    
    # Quality score: Gaussian-like curve peaked at medium distances
    optimal_distance = 0.5
    quality_scores = torch.exp(-alpha * (normalized_dist - optimal_distance) ** 2)
    
    # Penalize very similar patches
    too_similar_penalty = torch.exp(-10 * normalized_dist)
    quality_scores = quality_scores * (1 - 0.5 * too_similar_penalty)
    
    # Normalize to get sampling probabilities
    weights = quality_scores / (quality_scores.sum(dim=-1, keepdim=True) + 1e-8)
    
    return weights
