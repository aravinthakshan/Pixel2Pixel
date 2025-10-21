import subprocess
import argparse

parser = argparse.ArgumentParser('Pixel2Pixel')
parser.add_argument('--dataset', default='kodak', type=str, help='Dataset name')
parser.add_argument('--nt', default='gauss', type=str, help='Noise type')
args = parser.parse_args()

noise_levels = [10, 25, 50]

for nl in noise_levels:
    print(f"\n=== NOISE TYPE: {args.noise_type.upper()} | NOISE LEVEL: {nl} ===\n")
    cmd = [
        "python","train.py",
        "--num_iterations","1",
        "--nn_layers","6",
        "--epochs_per_iter","3000",
        "--gt_dir","GT",
        "--noisy_dir","Noisy",
        "--nt",args.noise_type,
        "--dataset",args.dataset,
        "--loss","L1",
        "--use_quality_weights","True",
        "--nl",str(nl)
    ]
    subprocess.run(cmd)