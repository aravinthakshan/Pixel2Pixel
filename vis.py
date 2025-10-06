import matplotlib.pyplot as plt
from Pixel2Pixel_syn import *

def visualize_denoising(sample_idx=0):
    """
    Visualize clean, noisy, and denoised outputs for a given image index.
    
    Args:
        sample_idx (int): Index of the image in the dataset directory.
    """
    image_folder = os.path.join(args.data_path, args.dataset)
    image_files = sorted(os.listdir(image_folder))
    
    if sample_idx >= len(image_files):
        print(f"Invalid index {sample_idx}. Max available = {len(image_files)-1}")
        return
    
    image_file = image_files[sample_idx]
    image_path = os.path.join(image_folder, image_file)

    # Load clean image
    clean_img = Image.open(image_path).convert("RGB")
    clean_tensor = transform(clean_img).unsqueeze(0).to(device)

    # Add noise
    noisy_tensor = add_noise(clean_tensor, args.nl)
    
    # Denoise using your trained model
    model.eval()
    with torch.no_grad():
        denoised_tensor = torch.clamp(model(noisy_tensor), 0, 1)

    # Convert tensors to numpy for visualization
    clean_np = clean_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    noisy_np = noisy_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    denoised_np = denoised_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(clean_np)
    plt.title("Clean Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_np)
    plt.title(f"Noisy ({args.nt}, {args.nl})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(denoised_np)
    plt.title("Denoised Output")
    plt.axis("off")

    plt.suptitle(f"Sample: {image_file}", fontsize=14)
    plt.tight_layout()
    plt.show()
