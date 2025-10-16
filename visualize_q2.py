import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch
import argparse

class TrainingPairsViewer:
    def __init__(self, bank_path, distances_path=None, alpha=2.0):
        """
        Interactive viewer for training pairs from pixel bank.
        
        Args:
            bank_path: Path to .npy file containing pixel bank
            distances_path: Path to distances file (optional, for weighted sampling)
            alpha: Alpha parameter for quality weighting
        """
        # Load pixel bank
        print(f"Loading pixel bank from: {bank_path}")
        # bank_arr = np.load(bank_path)
        bank_arr = np.load(bank_path, allow_pickle=True)

        
        if bank_arr.ndim == 3:
            bank_arr = np.expand_dims(bank_arr, axis=-1)
        
        # Transpose to (H, W, K, C)
        if bank_arr.shape[0] < bank_arr.shape[2]:  # If it's (K, C, H, W)
            self.img_bank = bank_arr.transpose((2, 3, 0, 1))
        else:  # Already (H, W, K, C)
            self.img_bank = bank_arr
            
        self.H, self.W, self.K, self.C = self.img_bank.shape
        print(f"Bank shape: (H={self.H}, W={self.W}, K={self.K}, C={self.C})")
        
        # Convert to torch for sampling
        self.img_bank_tensor = torch.from_numpy(self.img_bank.astype(np.float32))
        
        # Load distances and compute weights if available
        self.weights = None
        self.distances = None
        if distances_path and os.path.exists(distances_path):
            print(f"Loading distances from: {distances_path}")
            dist_arr = np.load(distances_path)
            self.distances = torch.from_numpy(dist_arr.astype(np.float32))
            self.weights = self.compute_quality_weights(self.distances, alpha)
            print(f"Quality-weighted sampling enabled (α={alpha})")
        else:
            print("Using uniform sampling")
        
        self.alpha = alpha
        self.current_pair_idx = 0
        self.pairs_cache = []
        self.cache_size = 100  # Pre-generate pairs
        
        # Generate initial pairs
        self.generate_pairs(self.cache_size)
        
        # Setup plot
        self.setup_plot()
    
    def compute_quality_weights(self, distances, alpha=2.0):
        """Compute quality weights from distances."""
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
    
    def generate_pairs(self, num_pairs):
        """Generate training pairs like the actual training loop."""
        print(f"Generating {num_pairs} training pairs...")
        
        for _ in range(num_pairs):
            # Random pixel location
            py = np.random.randint(0, self.H)
            px = np.random.randint(0, self.W)
            
            # Sample two bank indices
            if self.weights is not None:
                # Weighted sampling
                pixel_weights = self.weights[py, px]
                idx1 = torch.multinomial(pixel_weights, num_samples=1).item()
                idx2 = torch.multinomial(pixel_weights, num_samples=1).item()
                while idx2 == idx1 and self.K > 1:
                    idx2 = torch.multinomial(pixel_weights, num_samples=1).item()
                
                weight1 = pixel_weights[idx1].item()
                weight2 = pixel_weights[idx2].item()
                dist1 = self.distances[py, px, idx1].item() if self.distances is not None else None
                dist2 = self.distances[py, px, idx2].item() if self.distances is not None else None
            else:
                # Uniform sampling
                idx1 = np.random.randint(0, self.K)
                idx2 = np.random.randint(0, self.K)
                while idx2 == idx1 and self.K > 1:
                    idx2 = np.random.randint(0, self.K)
                
                weight1 = 1.0 / self.K
                weight2 = 1.0 / self.K
                dist1 = None
                dist2 = None
            
            # Get pixel values
            pixel1 = self.img_bank[py, px, idx1]
            pixel2 = self.img_bank[py, px, idx2]
            
            # Store pair info
            pair_info = {
                'pixel1': pixel1,
                'pixel2': pixel2,
                'location': (py, px),
                'idx1': idx1,
                'idx2': idx2,
                'weight1': weight1,
                'weight2': weight2,
                'dist1': dist1,
                'dist2': dist2
            }
            self.pairs_cache.append(pair_info)
    
    def setup_plot(self):
        """Setup the interactive matplotlib plot."""
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.suptitle('Training Pairs Viewer - Click "Next" to see pairs', 
                         fontsize=16, fontweight='bold')
        
        # Create axes for two images and info
        self.ax1 = plt.subplot(2, 3, 1)
        self.ax2 = plt.subplot(2, 3, 2)
        self.ax_info = plt.subplot(2, 3, 3)
        
        # Create axes for pixel value comparison
        self.ax_values = plt.subplot(2, 3, (4, 6))
        
        # Remove ticks from info axis
        self.ax_info.axis('off')
        
        # Add navigation buttons
        ax_prev = plt.axes([0.3, 0.02, 0.1, 0.05])
        ax_next = plt.axes([0.6, 0.02, 0.1, 0.05])
        ax_rand = plt.axes([0.45, 0.02, 0.1, 0.05])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_rand = Button(ax_rand, 'Random')
        
        self.btn_prev.on_clicked(self.prev_pair)
        self.btn_next.on_clicked(self.next_pair)
        self.btn_rand.on_clicked(self.random_pair)
        
        # Display first pair
        self.display_pair()
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        plt.show()
    
    def display_pair(self):
        """Display current training pair."""
        pair = self.pairs_cache[self.current_pair_idx]
        
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax_info.clear()
        self.ax_values.clear()
        self.ax_info.axis('off')
        
        pixel1 = pair['pixel1']
        pixel2 = pair['pixel2']
        py, px = pair['location']
        
        # Display pixel 1 (Input 1)
        if self.C == 3:
            img1 = np.clip(pixel1, 0, 1)
        else:
            img1 = np.clip(pixel1[0], 0, 1)
            img1 = np.stack([img1, img1, img1], axis=-1)
        
        self.ax1.imshow([[img1]], aspect='auto')
        self.ax1.set_title(f'Input 1\n(Bank Index: {pair["idx1"]})', 
                          fontsize=14, fontweight='bold', color='blue')
        self.ax1.axis('off')
        
        # Display pixel 2 (Input 2)
        if self.C == 3:
            img2 = np.clip(pixel2, 0, 1)
        else:
            img2 = np.clip(pixel2[0], 0, 1)
            img2 = np.stack([img2, img2, img2], axis=-1)
        
        self.ax2.imshow([[img2]], aspect='auto')
        self.ax2.set_title(f'Input 2\n(Bank Index: {pair["idx2"]})', 
                          fontsize=14, fontweight='bold', color='green')
        self.ax2.axis('off')
        
        # Display info
        info_text = f"PAIR {self.current_pair_idx + 1} / {len(self.pairs_cache)}\n"
        info_text += "=" * 35 + "\n\n"
        info_text += f"Pixel Location: ({py}, {px})\n\n"
        
        info_text += "INPUT 1:\n"
        info_text += f"  Bank Index: {pair['idx1']}\n"
        if self.C == 3:
            info_text += f"  RGB: ({pixel1[0]:.3f}, {pixel1[1]:.3f}, {pixel1[2]:.3f})\n"
        else:
            info_text += f"  Value: {pixel1[0]:.3f}\n"
        info_text += f"  Weight: {pair['weight1']:.4f}\n"
        if pair['dist1'] is not None:
            info_text += f"  Distance: {pair['dist1']:.4f}\n"
        
        info_text += "\nINPUT 2:\n"
        info_text += f"  Bank Index: {pair['idx2']}\n"
        if self.C == 3:
            info_text += f"  RGB: ({pixel2[0]:.3f}, {pixel2[1]:.3f}, {pixel2[2]:.3f})\n"
        else:
            info_text += f"  Value: {pixel2[0]:.3f}\n"
        info_text += f"  Weight: {pair['weight2']:.4f}\n"
        if pair['dist2'] is not None:
            info_text += f"  Distance: {pair['dist2']:.4f}\n"
        
        info_text += "\nDIFFERENCE:\n"
        pixel_diff = np.abs(pixel1 - pixel2).mean()
        info_text += f"  Mean Abs Diff: {pixel_diff:.4f}\n"
        
        if self.weights is not None:
            info_text += f"\nSampling: Quality-weighted (α={self.alpha})"
        else:
            info_text += f"\nSampling: Uniform"
        
        self.ax_info.text(0.05, 0.95, info_text, 
                         transform=self.ax_info.transAxes,
                         fontsize=11, verticalalignment='top',
                         family='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Plot pixel values comparison
        if self.C == 3:
            x = ['R', 'G', 'B']
            width = 0.35
            x_pos = np.arange(len(x))
            
            self.ax_values.bar(x_pos - width/2, pixel1, width, 
                              label='Input 1', alpha=0.8, color='blue')
            self.ax_values.bar(x_pos + width/2, pixel2, width, 
                              label='Input 2', alpha=0.8, color='green')
            
            self.ax_values.set_ylabel('Pixel Value', fontsize=12)
            self.ax_values.set_xlabel('Channel', fontsize=12)
            self.ax_values.set_title('Pixel Value Comparison', fontsize=13, fontweight='bold')
            self.ax_values.set_xticks(x_pos)
            self.ax_values.set_xticklabels(x)
            self.ax_values.legend()
            self.ax_values.set_ylim([0, 1])
            self.ax_values.grid(True, alpha=0.3, axis='y')
        else:
            self.ax_values.bar([0, 1], [pixel1[0], pixel2[0]], 
                              color=['blue', 'green'], alpha=0.8)
            self.ax_values.set_ylabel('Pixel Value', fontsize=12)
            self.ax_values.set_title('Pixel Value Comparison', fontsize=13, fontweight='bold')
            self.ax_values.set_xticks([0, 1])
            self.ax_values.set_xticklabels(['Input 1', 'Input 2'])
            self.ax_values.set_ylim([0, 1])
            self.ax_values.grid(True, alpha=0.3, axis='y')
        
        plt.draw()
    
    def next_pair(self, event):
        """Show next pair."""
        self.current_pair_idx = (self.current_pair_idx + 1) % len(self.pairs_cache)
        
        # Generate more pairs if running low
        if self.current_pair_idx > len(self.pairs_cache) - 10:
            self.generate_pairs(50)
        
        self.display_pair()
    
    def prev_pair(self, event):
        """Show previous pair."""
        self.current_pair_idx = (self.current_pair_idx - 1) % len(self.pairs_cache)
        self.display_pair()
    
    def random_pair(self, event):
        """Show random pair."""
        self.current_pair_idx = np.random.randint(0, len(self.pairs_cache))
        self.display_pair()


def main():
    parser = argparse.ArgumentParser(description='Interactive Training Pairs Viewer')
    parser.add_argument('--bank_path', type=str, required=True,
                       help='Path to pixel bank .npy file')
    parser.add_argument('--distances_path', type=str, default=None,
                       help='Path to distances .npy file (optional, for weighted sampling)')
    parser.add_argument('--alpha', type=float, default=2.0,
                       help='Alpha parameter for quality weighting')
    
    args = parser.parse_args()
    
    # Auto-detect distances file if not provided
    if args.distances_path is None:
        potential_dist_path = args.bank_path.replace('.npy', '_distances.npy')
        if os.path.exists(potential_dist_path):
            args.distances_path = potential_dist_path
            print(f"Auto-detected distances file: {potential_dist_path}")
    
    print("\n" + "="*60)
    print("INTERACTIVE TRAINING PAIRS VIEWER")
    print("="*60)
    print("\nControls:")
    print("  - Click 'Next' to see next training pair")
    print("  - Click 'Previous' to see previous pair")
    print("  - Click 'Random' to jump to random pair")
    print("  - Close window to exit")
    print("="*60 + "\n")
    
    viewer = TrainingPairsViewer(
        bank_path=args.bank_path,
        distances_path=args.distances_path,
        alpha=args.alpha
    )


if __name__ == "__main__":
    # Example usage without command line args (for testing)
    import sys
    
    if len(sys.argv) == 1:
        print("Usage:")
        print("  python interactive_viewer.py --bank_path path/to/kodim01.npy")
        print("\nOptional arguments:")
        print("  --distances_path path/to/kodim01_distances.npy")
        print("  --alpha 2.0")
        print("\nExample:")
        print("  python interactive_viewer.py --bank_path results/kodak_gauss_50.0_40_7_16_L1/kodim01.npy --alpha 2.0")
    else:
        main()