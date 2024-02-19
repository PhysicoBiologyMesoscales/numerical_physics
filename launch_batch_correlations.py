import os
import subprocess

if __name__ == "__main__":
    base_path = r"C:\Users\nolan\Documents\PhD\Simulations\Data\Circle_Meeting\Batch"
    for folder in os.listdir(base_path):
        data_path = os.path.join(base_path, folder)
        if "correlations.npy" in os.listdir(data_path):
            continue
        print("Computing correlations for file " + folder)
        subprocess.run(f"python batch_compute_correlations.py {data_path}")
