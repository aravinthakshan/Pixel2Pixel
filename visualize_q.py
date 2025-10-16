import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from matplotlib.gridspec import GridSpec
import argparse

def compute_quality_weights_distance(distances, alpha=2.0, optimal_distance=0.5):
    """
    Compute quality weights from distances with configurable parameters.
    """
    # Normalize distances per pixel to [0, 1]
    dist_min = distances.min(dim=-1, keepdim=True)[0]
    dist_max = distances.max(dim=-1, keepdim=True)[0]
    dist_range = dist_max - dist_min
    dist_range = torch.clamp(dist_range, min=1e-8)
    normalized_dist = (distances - dist_min) / dist_range
    
    # Quality score: Gaussian-like curve peaked at optimal distance
    quality_scores = torch.exp(-alpha * (normalized_dist - optimal_distance) ** 2)
    
    # Penalize very similar patches
    too_similar_penalty = torch.exp(-10 * normalized_dist)
    quality_scores = quality_scores * (1 - 0.5 * too_similar_penalty)
    
    # Normalize to get sampling probabilities
    weights = quality_scores / (quality_scores.sum(dim=-1, keepdim=True) + 1e-8)
    
    return weights, normalized_dist, quality_scores


def plot_weight_functions(alphas=[0.5, 1.0, 2.0, 5.0, 10.0], 
                          optimal_distances=[0.3, 0.5, 0.7],
                          save_path='weight_functions.png'):
    """
    Plot how weight function changes with different alpha and optimal_distance values.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Create normalized distance range
    normalized_dist = torch.linspace(0, 1, 1000)
    
    # Plot 1: Effect of alpha (fixed optimal_distance=0.5)
    ax1 = fig.add_subplot(gs[0, :])
    optimal_dist = 0.5
    for alpha in alphas:
        quality_scores = torch.exp(-alpha * (normalized_dist - optimal_dist) ** 2)
        too_similar_penalty = torch.exp(-10 * normalized_dist)
        final_scores = quality_scores * (1 - 0.5 * too_similar_penalty)
        
        ax1.plot(normalized_dist.numpy(), final_scores.numpy(), 
                label=f'α={alpha}', linewidth=2)
    
    ax1.set_xlabel('Normalized Distance', fontsize=12)
    ax1.set_ylabel('Quality Score', fontsize=12)
    ax1.set_title(f'Effect of Alpha (α) on Quality Scores (optimal_distance={optimal_dist})', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=optimal_dist, color='red', linestyle='--', alpha=0.5, label='Optimal Distance')
    
    # Plot 2-4: Effect of optimal_distance for different alphas
    for idx, alpha in enumerate([1.0, 2.0, 5.0]):
        ax = fig.add_subplot(gs[1, idx])
        
        for opt_dist in optimal_distances:
            quality_scores = torch.exp(-alpha * (normalized_dist - opt_dist) ** 2)
            too_similar_penalty = torch.exp(-10 * normalized_dist)
            final_scores = quality_scores * (1 - 0.5 * too_similar_penalty)
            
            ax.plot(normalized_dist.numpy(), final_scores.numpy(), 
                   label=f'opt_dist={opt_dist}', linewidth=2)
        
        ax.set_xlabel('Normalized Distance', fontsize=10)
        ax.set_ylabel('Quality Score', fontsize=10)
        ax.set_title(f'α={alpha}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved weight functions plot to {save_path}")
    plt.close()


def visualize_sampling_distribution(bank_path, image_name, 
                                    alphas=[1.0, 2.0, 5.0],
                                    num_pixels_to_sample=100,
                                    save_dir='sampling_viz'):
    """
    Visualize what samples are selected with different alpha values.
    Shows the distribution of selected distances and sample indices.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load pixel bank and distances
    distances_path = bank_path.replace('.npy', '_distances.npy')
    
    if not os.path.exists(distances_path):
        print(f"Distance file not found: {distances_path}")
        return
    
    distances_arr = np.load(distances_path)
    distances = torch.from_numpy(distances_arr.astype(np.float32))
    
    print(f"Distances shape: {distances.shape}")  # Should be (H, W, K)
    H, W, K = distances.shape
    
    # Randomly select some pixels to analyze
    pixel_indices = torch.randint(0, H*W, (num_pixels_to_sample,))
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, len(alphas), figure=fig, hspace=0.4, wspace=0.3)
    
    for alpha_idx, alpha in enumerate(alphas):
        # Compute weights for all pixels
        weights, normalized_dist, quality_scores = compute_quality_weights_distance(
            distances, alpha=alpha
        )
        
        # Flatten for sampling
        weights_flat = weights.view(-1, K)
        distances_flat = distances.view(-1, K)
        normalized_dist_flat = normalized_dist.view(-1, K)
        
        # Sample from selected pixels
        sampled_indices = []
        sampled_distances = []
        sampled_normalized_distances = []
        
        for pix_idx in pixel_indices:
            # Sample one index based on weights
            sampled_idx = torch.multinomial(weights_flat[pix_idx], num_samples=1)
            sampled_indices.append(sampled_idx.item())
            sampled_distances.append(distances_flat[pix_idx, sampled_idx].item())
            sampled_normalized_distances.append(normalized_dist_flat[pix_idx, sampled_idx].item())
        
        # Plot 1: Histogram of sampled bank indices
        ax1 = fig.add_subplot(gs[0, alpha_idx])
        ax1.hist(sampled_indices, bins=min(K, 30), edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Bank Index (k)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title(f'α={alpha}\nSampled Bank Indices', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Histogram of sampled normalized distances
        ax2 = fig.add_subplot(gs[1, alpha_idx])
        ax2.hist(sampled_normalized_distances, bins=30, edgecolor='black', alpha=0.7, color='orange')
        ax2.set_xlabel('Normalized Distance', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('Sampled Distances', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Optimal (0.5)')
        ax2.legend()
        
        # Plot 3: Weight distribution vs distance for a few example pixels
        ax3 = fig.add_subplot(gs[2, alpha_idx])
        for i in range(min(5, len(pixel_indices))):
            pix_idx = pixel_indices[i]
            ax3.plot(normalized_dist_flat[pix_idx].numpy(), 
                    weights_flat[pix_idx].numpy(),
                    alpha=0.6, marker='o', markersize=3)
        
        ax3.set_xlabel('Normalized Distance', fontsize=10)
        ax3.set_ylabel('Sampling Weight', fontsize=10)
        ax3.set_title('Weight vs Distance\n(5 example pixels)', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, f'{image_name}_sampling_distribution.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved sampling distribution plot to {save_dir}/{image_name}_sampling_distribution.png")
    plt.close()


def visualize_training_pairs(bank_path, image_name, alpha=2.0, 
                             num_examples=5, save_dir='training_pairs_viz'):
    """
    Visualize what the model sees during training:
    - Show actual pixel patches being paired
    - Display their distances and weights
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load pixel bank and distances
    bank_arr = np.load(bank_path)
    distances_path = bank_path.replace('.npy', '_distances.npy')
    
    if not os.path.exists(distances_path):
        print(f"Distance file not found: {distances_path}")
        return
    
    distances_arr = np.load(distances_path)
    
    # Convert to torch
    img_bank = torch.from_numpy(bank_arr.astype(np.float32))
    distances = torch.from_numpy(distances_arr.astype(np.float32))
    
    print(f"Bank shape: {img_bank.shape}")  # Should be (H, W, K, C)
    print(f"Distances shape: {distances.shape}")  # Should be (H, W, K)
    
    if img_bank.ndim == 3:
        img_bank = img_bank.unsqueeze(-1)  # Add channel dimension if grayscale
    
    H, W, K, C = img_bank.shape
    
    # Compute weights
    weights, normalized_dist, _ = compute_quality_weights_distance(distances, alpha=alpha)
    
    # Select random pixel locations
    pixel_ys = torch.randint(0, H, (num_examples,))
    pixel_xs = torch.randint(0, W, (num_examples,))
    
    fig, axes = plt.subplots(num_examples, 6, figsize=(20, 4*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_examples):
        py, px = pixel_ys[i].item(), pixel_xs[i].item()
        
        # Get the pixel bank for this location
        pixel_banks = img_bank[py, px]  # Shape: (K, C)
        pixel_weights = weights[py, px]  # Shape: (K,)
        pixel_distances = distances[py, px]  # Shape: (K,)
        pixel_norm_dist = normalized_dist[py, px]  # Shape: (K,)
        
        # Sample two indices based on weights
        idx1 = torch.multinomial(pixel_weights, num_samples=1).item()
        idx2 = torch.multinomial(pixel_weights, num_samples=1).item()
        while idx2 == idx1 and K > 1:
            idx2 = torch.multinomial(pixel_weights, num_samples=1).item()
        
        # Get the pixel values
        pixel1 = pixel_banks[idx1].numpy()
        pixel2 = pixel_banks[idx2].numpy()
        
        # Display pixel 1 (as color if RGB)
        ax = axes[i, 0]
        if C == 3:
            color1 = np.clip(pixel1, 0, 1)
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=color1))
        else:
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=[pixel1[0]]*3))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'Pixel 1 (bank {idx1})', fontsize=10)
        
        # Display pixel 2
        ax = axes[i, 1]
        if C == 3:
            color2 = np.clip(pixel2, 0, 1)
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=color2))
        else:
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=[pixel2[0]]*3))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'Pixel 2 (bank {idx2})', fontsize=10)
        
        # Show info about the pair
        ax = axes[i, 2]
        ax.axis('off')
        info_text = (
            f"Location: ({py}, {px})\n\n"
            f"Pixel 1:\n"
            f"  Bank index: {idx1}\n"
            f"  Distance: {pixel_distances[idx1]:.4f}\n"
            f"  Norm dist: {pixel_norm_dist[idx1]:.4f}\n"
            f"  Weight: {pixel_weights[idx1]:.4f}\n\n"
            f"Pixel 2:\n"
            f"  Bank index: {idx2}\n"
            f"  Distance: {pixel_distances[idx2]:.4f}\n"
            f"  Norm dist: {pixel_norm_dist[idx2]:.4f}\n"
            f"  Weight: {pixel_weights[idx2]:.4f}\n\n"
            f"Pixel diff: {np.abs(pixel1 - pixel2).mean():.4f}"
        )
        ax.text(0.1, 0.5, info_text, fontsize=9, verticalalignment='center',
                family='monospace')
        
        # Plot weight distribution for this pixel
        ax = axes[i, 3]
        bars = ax.bar(range(K), pixel_weights.numpy(), alpha=0.7)
        bars[idx1].set_color('red')
        bars[idx2].set_color('blue')
        ax.set_xlabel('Bank Index', fontsize=9)
        ax.set_ylabel('Weight', fontsize=9)
        ax.set_title('Sampling Weights\n(red=pix1, blue=pix2)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot distances
        ax = axes[i, 4]
        ax.plot(range(K), pixel_norm_dist.numpy(), 'o-', alpha=0.7)
        ax.plot(idx1, pixel_norm_dist[idx1].item(), 'ro', markersize=10, label='Pixel 1')
        ax.plot(idx2, pixel_norm_dist[idx2].item(), 'bo', markersize=10, label='Pixel 2')
        ax.set_xlabel('Bank Index', fontsize=9)
        ax.set_ylabel('Normalized Distance', fontsize=9)
        ax.set_title('Distances from Reference', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Show all bank values for this pixel
        ax = axes[i, 5]
        bank_values = pixel_banks.numpy()
        if C == 3:
            # Show RGB values
            colors = ['red', 'green', 'blue']
            for c, (color_name, color_val) in enumerate(zip(['R', 'G', 'B'], colors)):
                ax.plot(range(K), bank_values[:, c], 'o-', alpha=0.7, label=color_name, color=color_val)
                # Highlight selected samples for this channel
                ax.plot(idx1, bank_values[idx1, c], 'o', markersize=12, 
                       markeredgecolor='black', markeredgewidth=2, color=color_val, alpha=0.8)
                ax.plot(idx2, bank_values[idx2, c], 's', markersize=12, 
                       markeredgecolor='black', markeredgewidth=2, color=color_val, alpha=0.8)
        else:
            ax.plot(range(K), bank_values[:, 0], 'o-', alpha=0.7)
            ax.plot(idx1, bank_values[idx1, 0], 'ro', markersize=10)
            ax.plot(idx2, bank_values[idx2, 0], 'bs', markersize=10)
        ax.set_xlabel('Bank Index', fontsize=9)
        ax.set_ylabel('Pixel Value', fontsize=9)
        ax.set_title('Bank Pixel Values\n(○=pix1, □=pix2)', fontsize=10)
        if C == 3:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, f'{image_name}_training_pairs_alpha{alpha}.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved training pairs visualization to {save_dir}/{image_name}_training_pairs_alpha{alpha}.png")
    plt.close()


def compare_uniform_vs_weighted_sampling(bank_path, image_name, alpha=2.0,
                                        num_samples=1000, save_dir='sampling_comparison'):
    """
    Compare uniform sampling vs weighted sampling statistics.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    distances_path = bank_path.replace('.npy', '_distances.npy')
    if not os.path.exists(distances_path):
        print(f"Distance file not found: {distances_path}")
        return
    
    distances_arr = np.load(distances_path)
    distances = torch.from_numpy(distances_arr.astype(np.float32))
    H, W, K = distances.shape
    
    # Compute weights
    weights, normalized_dist, _ = compute_quality_weights_distance(distances, alpha=alpha)
    
    # Flatten
    weights_flat = weights.view(-1, K)
    distances_flat = distances.view(-1, K)
    normalized_dist_flat = normalized_dist.view(-1, K)
    
    # Sample pixels to test
    num_test_pixels = min(100, H * W)
    test_pixel_indices = torch.randperm(H * W)[:num_test_pixels]
    
    # Collect samples
    uniform_samples = []
    weighted_samples = []
    uniform_distances = []
    weighted_distances = []
    
    for pix_idx in test_pixel_indices:
        for _ in range(num_samples // num_test_pixels):
            # Uniform sampling
            uniform_idx = torch.randint(0, K, (1,)).item()
            uniform_samples.append(uniform_idx)
            uniform_distances.append(normalized_dist_flat[pix_idx, uniform_idx].item())
            
            # Weighted sampling
            weighted_idx = torch.multinomial(weights_flat[pix_idx], num_samples=1).item()
            weighted_samples.append(weighted_idx)
            weighted_distances.append(normalized_dist_flat[pix_idx, weighted_idx].item())
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Bank index distribution
    ax = axes[0, 0]
    ax.hist(uniform_samples, bins=K, alpha=0.5, label='Uniform', edgecolor='black')
    ax.hist(weighted_samples, bins=K, alpha=0.5, label=f'Weighted (α={alpha})', edgecolor='black')
    ax.set_xlabel('Bank Index', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Bank Index Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distance distribution
    ax = axes[0, 1]
    ax.hist(uniform_distances, bins=50, alpha=0.5, label='Uniform', edgecolor='black', density=True)
    ax.hist(weighted_distances, bins=50, alpha=0.5, label=f'Weighted (α={alpha})', edgecolor='black', density=True)
    ax.set_xlabel('Normalized Distance', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Sampled Distance Distribution', fontsize=12, fontweight='bold')
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Optimal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Statistics box
    ax = axes[1, 0]
    ax.axis('off')
    stats_text = (
        f"Sampling Statistics ({num_samples} samples)\n"
        f"{'='*45}\n\n"
        f"Uniform Sampling:\n"
        f"  Mean distance: {np.mean(uniform_distances):.4f}\n"
        f"  Std distance: {np.std(uniform_distances):.4f}\n"
        f"  Median distance: {np.median(uniform_distances):.4f}\n\n"
        f"Weighted Sampling (α={alpha}):\n"
        f"  Mean distance: {np.mean(weighted_distances):.4f}\n"
        f"  Std distance: {np.std(weighted_distances):.4f}\n"
        f"  Median distance: {np.median(weighted_distances):.4f}\n\n"
        f"Difference:\n"
        f"  Δ Mean: {np.mean(weighted_distances) - np.mean(uniform_distances):.4f}\n"
        f"  Shift toward optimal: {abs(np.mean(weighted_distances) - 0.5) < abs(np.mean(uniform_distances) - 0.5)}"
    )
    ax.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
            family='monospace')
    
    # CDF comparison
    ax = axes[1, 1]
    uniform_sorted = np.sort(uniform_distances)
    weighted_sorted = np.sort(weighted_distances)
    uniform_cdf = np.arange(1, len(uniform_sorted) + 1) / len(uniform_sorted)
    weighted_cdf = np.arange(1, len(weighted_sorted) + 1) / len(weighted_sorted)
    
    ax.plot(uniform_sorted, uniform_cdf, label='Uniform', linewidth=2)
    ax.plot(weighted_sorted, weighted_cdf, label=f'Weighted (α={alpha})', linewidth=2)
    ax.set_xlabel('Normalized Distance', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{image_name}_uniform_vs_weighted_alpha{alpha}.png'),
                dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {save_dir}/{image_name}_uniform_vs_weighted_alpha{alpha}.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize quality-based sampling')
    parser.add_argument('--bank_dir', type=str, required=True, 
                       help='Directory containing pixel banks (e.g., results/kodak_gauss_50.0_40_7_16_L1)')
    parser.add_argument('--image_name', type=str, default='kodim01',
                       help='Image name without extension (e.g., kodim01)')
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.5, 1.0, 2.0, 5.0],
                       help='Alpha values to test')
    parser.add_argument('--output_dir', type=str, default='visualization_output',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate weight function plots
    print("\n" + "="*60)
    print("Generating weight function plots...")
    print("="*60)
    plot_weight_functions(
        alphas=args.alphas,
        save_path=os.path.join(args.output_dir, 'weight_functions.png')
    )
    
    # Path to pixel bank
    bank_path = os.path.join(args.bank_dir, f'{args.image_name}.npy')
    
    if os.path.exists(bank_path):
        # Generate sampling distribution plots
        print("\n" + "="*60)
        print("Generating sampling distribution visualizations...")
        print("="*60)
        visualize_sampling_distribution(
            bank_path, 
            args.image_name,
            alphas=args.alphas,
            save_dir=os.path.join(args.output_dir, 'sampling_distribution')
        )
        
        # Generate training pairs visualizations for each alpha
        for alpha in args.alphas:
            print("\n" + "="*60)
            print(f"Generating training pairs visualization for α={alpha}...")
            print("="*60)
            visualize_training_pairs(
                bank_path,
                args.image_name,
                alpha=alpha,
                num_examples=5,
                save_dir=os.path.join(args.output_dir, 'training_pairs')
            )
        
        # Generate comparison plots
        for alpha in args.alphas:
            print("\n" + "="*60)
            print(f"Generating uniform vs weighted comparison for α={alpha}...")
            print("="*60)
            compare_uniform_vs_weighted_sampling(
                bank_path,
                args.image_name,
                alpha=alpha,
                save_dir=os.path.join(args.output_dir, 'sampling_comparison')
            )
    else:
        print(f"\nError: Bank file not found at {bank_path}")
        print("Only generating weight function plots.")
    
    print("\n" + "="*60)
    print("Visualization complete! Check the output directory:")
    print(f"  {args.output_dir}/")
    print("="*60)