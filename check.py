# import numpy as np
# from PIL import Image
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# def calculate_psnr_from_paths(img_path1, img_path2):
#     """
#     Compute PSNR between two images given their file paths.
    
#     Args:
#         img_path1 (str): Path to the first image (e.g., reference/ground truth)
#         img_path2 (str): Path to the second image (e.g., reconstructed/denoised)
    
#     Returns:
#         float: PSNR value in decibels (dB)
#     """
#     # Load images
#     img1 = np.array(Image.open(img_path1).convert('RGB'), dtype=np.float32)
#     img2 = np.array(Image.open(img_path2).convert('RGB'), dtype=np.float32)

#     # Ensure same size
#     if img1.shape != img2.shape:
#         raise ValueError(f"Image shapes do not match: {img1.shape} vs {img2.shape}")

#     # Compute PSNR (max pixel value = 255 for 8-bit images)
#     psnr_value = compare_psnr(img1, img2, data_range=255.0)
#     return psnr_value


# # Example usage:
# psnr = calculate_psnr_from_paths("/home/aravinthakshan/Projects/Research/Audio_Denoising/Pixel2Pixel/8_noisy.png", "/home/aravinthakshan/Projects/Research/Audio_Denoising/Pixel2Pixel/8.png")
# print(f"PSNR: {psnr:.2f} dB")
