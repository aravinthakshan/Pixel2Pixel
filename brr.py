import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def center_crop_images(src_dir, dst_dir, crop_size=256):
    """
    Center-crop all images in src_dir to crop_size x crop_size
    and save them to dst_dir.

    Args:
        src_dir (str): Source directory containing input images.
        dst_dir (str): Destination directory to save cropped images.
        crop_size (int): Size for center crop (default=256).
    """
    os.makedirs(dst_dir, exist_ok=True)
    transform = transforms.CenterCrop(crop_size)

    image_files = [f for f in os.listdir(src_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    for img_file in tqdm(image_files, desc="Center cropping images"):
        img_path = os.path.join(src_dir, img_file)
        out_path = os.path.join(dst_dir, img_file)

        try:
            img = Image.open(img_path).convert("RGB")
            cropped = transform(img)
            cropped.save(out_path)
        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")

    print(f"\n✅ Saved {len(image_files)} cropped images to: {dst_dir}")

# Example usage
if __name__ == "__main__":
    src = "/home/aravinthakshan/Projects/Research/Audio_Denoising/Pixel2Pixel/data/McMaster"
    dst = "/home/aravinthakshan/Projects/Research/Audio_Denoising/Pixel2Pixel/data/mcmaster"
    center_crop_images(src, dst, crop_size=256)
