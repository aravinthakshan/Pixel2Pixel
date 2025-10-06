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
parser.add_argument('--dataset', default='SIDD', type=str, help='Dataset name')
parser.add_argument('--GT', default='GT', type=str, help='Folder name for ground truth images')
parser.add_argument('--Noisy', default='Noisy', type=str, help='Folder name for noisy images')
parser.add_argument('--save', default='./results', type=str, help='Directory to save pixel bank results')
parser.add_argument('--out_image', default='./results_image', type=str, help='Directory to save denoised images')
parser.add_argument('--ws', default=40, type=int, help='Window size')
parser.add_argument('--ps', default=7, type=int, help='Patch size')
parser.add_argument('--nn', default=100, type=int, help='Number of nearest neighbors to search')
parser.add_argument('--mm', default=20, type=int, help='Number of pixel banks to use for training')
parser.add_argument('--loss', default='L2', type=str, help='Loss function type')
args = parser.parse_args()

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda:0"

WINDOW_SIZE = args.ws
PATCH_SIZE = args.ps
NUM_NEIGHBORS = args.nn
loss_type = args.loss

transform = transforms.Compose([transforms.ToTensor()])

def construct_pixel_bank():
    # The pixel banks will be saved in a directory constructed from dataset parameters
    bank_dir = os.path.join(args.save, args.dataset, '_'.join(str(i) for i in [args.ws, args.ps, args.nn, args.loss]))
    os.makedirs(bank_dir, exist_ok=True)

    noisy_folder = os.path.join(args.data_path, args.dataset, args.Noisy)
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

# -------------------------------
class DenoiseNet(nn.Module):
    def __init__(self, n_chan, chan_embed=64):
        super(DenoiseNet, self).__init__()
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
        return x

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


# -------------------------------
def train(model, optimizer, img_bank):
    N, H, W, C = img_bank.shape

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
# Denoising using the Constructed Pixel Bank
# -------------------------------
def denoise_images():
    # The pixel bank directory should match the one used in construction
    bank_dir = os.path.join(args.save, args.dataset, '_'.join(str(i) for i in [args.ws, args.ps, args.nn, args.loss]))
    gt_folder = os.path.join(args.data_path, args.dataset, args.GT)
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
        global model
        model = DenoiseNet(n_chan).to(device)
        print(f"Number of parameters for {image_file}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=[1500, 2000, 2500], gamma=0.5)

        for epoch in range(max_epoch):
            train(model, optimizer, img_bank)
            scheduler.step()

        PSNR, out_img = test(model, noisy_img, clean_img_tensor)
        out_img_pil = to_pil_image(out_img.squeeze(0))
        out_img_save_path = os.path.join(args.out_image, os.path.splitext(image_file)[0] + '.png')
        out_img_pil.save(out_img_save_path)

        noisy_img_pil = to_pil_image(noisy_img.squeeze(0))
        noisy_img_save_path = os.path.join(args.out_image, os.path.splitext(image_file)[0] + '_noisy.png')
        noisy_img_pil.save(noisy_img_save_path)

        out_img_loaded = io.imread(out_img_save_path)
        SSIM, _ = compare_ssim(clean_img_np, out_img_loaded, full=True, multichannel=True)
        print(f"Image: {image_file} | PSNR: {PSNR:.2f} dB | SSIM: {SSIM:.4f}")
        avg_PSNR += PSNR
        avg_SSIM += SSIM

    avg_PSNR /= len(gt_files)
    avg_SSIM /= len(gt_files)
    print(f"Average PSNR: {avg_PSNR:.2f} dB, Average SSIM: {avg_SSIM:.4f}")

if __name__ == "__main__":
    print("Constructing pixel banks from noisy images ...")
    construct_pixel_bank()
    print("Starting denoising using constructed pixel banks ...")
    denoise_images()
