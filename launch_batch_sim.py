import numpy as np
import subprocess
from itertools import product
from tqdm import tqdm
import os

if __name__ == "__main__":
    path_to_python_file = r"C:\Users\nolan\Documents\PhD\Simulations\numerical_physics\batch_simulation.py"
    aspect_ratio = 1.0
    N = 40000
    phi = 1.0
    kc = 3.0
    h = 0.0
    t_max = 100.0
    v_arr = np.linspace(1.6, 2.1, 10)
    k_arr = np.linspace(4 + 1 / 3, 5 - 1 / 3, 3)
    with tqdm(total=len(v_arr) * len(k_arr)) as progress_bar:
        for i, (v0, k) in enumerate(tqdm(product(v_arr, k_arr))):
            save_path = os.path.join(
                r"C:\Users\nolan\Documents\PhD\Simulations\\",
                "Data",
                "Force_computation",
                "Batch",
                f"ar={aspect_ratio}_N={N}_phi={phi}_v0={v0}_kc={kc}_k={k}_h={h}",
            )
            print(f"Launching simulation for v0={v0} and k={k}")
            if os.path.isdir(save_path):
                print("Simulation has already been run")
                progress_bar.update(1)  # update progress
                continue
            subprocess.run(
                f"python {path_to_python_file} {aspect_ratio} {N} {phi} {v0} {kc} {k} {h} {t_max} --save_images --save_path {save_path}"
            )
            progress_bar.update(1)  # update progress
