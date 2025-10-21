import subprocess

noise_levels = [10, 25, 50]
noise_types = ["gauss", "poiss"]

for nl in noise_levels:
    for nt in noise_types:
        cmd = [
            "python","train.py",
            "--num_iterations","1",
            "--nn_layers","6",
            "--epochs_per_iter","3000",
            "--gt_dir","GT",
            "--noisy_dir","Noisy",
            "--nt",nt,
            "--dataset","kodak",
            "--loss","L1",
            "--use_quality_weights","True",
            "--nl",str(nl)
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd)