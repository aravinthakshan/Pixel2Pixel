import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import einops

# Pixel bank construction function (from your code)
def construct_pixel_bank_from_image(img_tensor, file_name_without_ext, bank_dir):
    pad_sz = WINDOW_SIZE // 2 + PATCH_SIZE // 2
    center_offset = WINDOW_SIZE // 2
    blk_sz = 64

    img_pad = F.pad(img_tensor, (pad_sz, pad_sz, pad_sz, pad_sz), mode='reflect')
    img_unfold = F.unfold(img_pad, kernel_size=PATCH_SIZE, padding=0, stride=1)
    H_new = img_tensor.shape[-2] + WINDOW_SIZE
    W_new = img_tensor.shape[-1] + WINDOW_SIZE
    img_unfold = einops.rearrange(img_unfold, 'b c (h w) -> b c h w', h=H_new, w=W_new)

    num_blk_w = img_tensor.shape[-1] // blk_sz
    num_blk_h = img_tensor.shape[-2] // blk_sz
    is_window_size_even = (WINDOW_SIZE % 2 == 0)
    topk_list = []
    distance_list = []

    for blk_i in range(num_blk_w):
        for blk_j in range(num_blk_h):
            start_h = blk_j * blk_sz
            end_h = (blk_j + 1) * blk_sz + WINDOW_SIZE
            start_w = blk_i * blk_sz
            end_w = (blk_i + 1) * blk_sz + WINDOW_SIZE

            sub_img_uf = img_unfold[..., start_h:end_h, start_w:end_w]
            sub_img_shape = sub_img_uf.shape

            sub_img_uf_inp = sub_img_uf[..., :-1, :-1] if is_window_size_even else sub_img_uf
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

            distance = torch.sum((img_center - patch_windows) ** 2, dim=1) if LOSS_TYPE == 'L2' else torch.sum(torch.abs(img_center - patch_windows), dim=1)
            topk_distances, sort_indices = torch.topk(distance, k=NUM_NEIGHBORS, largest=False, sorted=True, dim=-3)

            patch_windows_reshape = einops.rearrange(
                patch_windows,
                'b (c k1 k2) (k3 k4) h w -> b c (k1 k2) (k3 k4) h w',
                k1=PATCH_SIZE, k2=PATCH_SIZE, k3=WINDOW_SIZE, k4=WINDOW_SIZE
            )
            patch_center = patch_windows_reshape[:, :, patch_windows_reshape.shape[2] // 2, ...]
            topk = torch.gather(patch_center, dim=-3, index=sort_indices.unsqueeze(1).repeat(1, 3, 1, 1, 1))
            topk_list.append(topk)
            distance_list.append(topk_distances)

    topk = torch.cat(topk_list, dim=0)
    topk = einops.rearrange(topk, '(w1 w2) c k h w -> k c (w2 h) (w1 w)', w1=num_blk_w, w2=num_blk_h)
    topk = topk.permute(2, 3, 0, 1)
    distances = torch.cat(distance_list, dim=0)
    distances = einops.rearrange(distances, '(w1 w2) k h w -> k (w2 h) (w1 w)', w1=num_blk_w, w2=num_blk_h)
    distances = distances.permute(1, 2, 0)

    np.save(os.path.join(bank_dir, file_name_without_ext + '.npy'), topk.cpu().numpy())
    np.save(os.path.join(bank_dir, file_name_without_ext + '_distances.npy'), distances.cpu().numpy())

    return topk, distances

def add_noise(x, noise_level, noise_type):
    device = x.device  # get the device of input tensor
    if noise_type == 'gauss':
        noisy = x + torch.normal(0, noise_level / 255, x.shape, device=device)
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


def visualize_pixel_bank(topk_path, distances_path, pixel_coord, save_path=None):
    topk = np.load(topk_path)
    distances = np.load(distances_path)
    h, w = pixel_coord
    K = topk.shape[2]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(3*(K+1), 3))

    center_patch = topk[h, w, 0]
    plt.subplot(1, K+1, 1)
    plt.imshow(center_patch[0] if center_patch.shape[0]==1 else np.transpose(center_patch, (1,2,0)),
               cmap='gray' if center_patch.shape[0]==1 else None)
    plt.title("Center pixel")

    for k in range(K):
        match_patch = topk[h, w, k]
        plt.subplot(1, K+1, k+2)
        plt.imshow(match_patch[0] if match_patch.shape[0]==1 else np.transpose(match_patch, (1,2,0)),
                   cmap='gray' if match_patch.shape[0]==1 else None)
        plt.title(f"Match {k}\nDist={distances[h,w,k]:.3f}")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='data/kodak/kodim01.png')
    parser.add_argument('--bank_dir', type=str, default='./pixel_bank')
    parser.add_argument('--noise_type', type=str, default='gauss')
    parser.add_argument('--noise_level', type=float, default=25)
    parser.add_argument('--pixel_h', type=int, default=0)
    parser.add_argument('--pixel_w', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=3)
    parser.add_argument('--window_size', type=int, default=7)
    parser.add_argument('--num_neighbors', type=int, default=5)
    parser.add_argument('--loss_type', type=str, default='L2')
    args = parser.parse_args()

    os.makedirs(args.bank_dir, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])

    PATCH_SIZE = args.patch_size
    WINDOW_SIZE = args.window_size
    NUM_NEIGHBORS = args.num_neighbors
    LOSS_TYPE = args.loss_type

    img = Image.open(args.image_path)
    img_tensor = transform(img).unsqueeze(0).cuda()[None, ...]
    img_tensor = add_noise(img_tensor, args.noise_level, args.noise_type)

    file_name_without_ext = os.path.splitext(os.path.basename(args.image_path))[0]
    topk, distances = construct_pixel_bank_from_image(img_tensor, file_name_without_ext, args.bank_dir)

    visualize_pixel_bank(
        os.path.join(args.bank_dir, file_name_without_ext + '.npy'),
        os.path.join(args.bank_dir, file_name_without_ext + '_distances.npy'),
        (args.pixel_h, args.pixel_w)
    )
