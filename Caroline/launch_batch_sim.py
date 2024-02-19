import numpy as np
import subprocess
from itertools import product
from tqdm import tqdm
import os

if __name__ == "__main__":
    path_to_python_file = r"C:\Users\nolan\Documents\PhD\Simulations\numerical_physics\Caroline\batch_simulation.py"
    aspect_ratio = 4.0
    N = 40000
    phi = 1.0
    kc = 3.0
    h_arr = np.linspace(0, 5, 10)
    h2_arr = np.linspace(0, 5, 10)
    t_max = 100.0
    v0 = 2
    k = 7
    with tqdm(total=len(h_arr) * len(h2_arr)) as progress_bar:
        for i, (h, h2) in enumerate(tqdm(product(h_arr, h2_arr))):
            save_path = os.path.join(
                r"C:\Users\nolan\Documents\PhD\Simulations\\",
                "Data",
                "Test_Caro",
                "Batch",
                f"ar={aspect_ratio}_N={N}_phi={phi}_v0={v0}_kc={kc}_k={k}_h={h}_h2={h2}",
            )
            print(f"Launching simulation for h={h} and h2={h2}")
            if os.path.isdir(save_path):
                print("Simulation has already been run")
                progress_bar.update(1)  # update progress
                continue
            subprocess.run(
                f"python {path_to_python_file} {aspect_ratio} {N} {phi} {v0} {kc} {k} {h} {h2} {t_max} --save_images --save_path {save_path}"
            )
            progress_bar.update(1)  # update progress
