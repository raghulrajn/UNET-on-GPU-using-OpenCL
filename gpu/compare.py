import numpy as np
import pandas as pd
import os
from PIL import Image

def compute_differences(pytorch, other, tolerance=0.00005):
    if pytorch.shape != other.shape:
        raise ValueError(f"Shapes do not match: {pytorch.shape} vs {other.shape}")
    absdiff = np.abs(pytorch - other)  
    return np.max(absdiff),np.average(absdiff), np.min(absdiff)

def get_common_suffixes(directory):
    files = os.listdir(directory)
    suffixes = [os.path.splitext(file)[0].split('_')[-1] for file in files if os.path.isfile(os.path.join(directory, file))]
    return list(set(suffixes))

results_folder = "results"
common_suffixes = get_common_suffixes(results_folder)

# Load data from .npy files with matching names
operations = ["convolution", "relu", "maxpool", "upsample", "batchnorm"]
images = ["img1", "img2", "img3"]
data = {}

for op in operations:
    data[f"gpu_{op}"] = np.load(os.path.join("npy",f"gpu_{op}.npy"))
    data[f"cpu_{op}"] = np.load(os.path.join("npy",f"cpu_{op}.npy"))
    data[f"pytorch_{op}"] = np.load(os.path.join("npy",f"pytorch_{op}.npy"))

# # Compute differences
results = []
for op in operations:
    #print(f"Comparing {op}...")
    max_diff_cpu,avg_diff_cpu, min_diff_cpu = compute_differences(data[f"pytorch_{op}"], data[f"cpu_{op}"])
    max_diff_gpu,avg_diff_gpu ,min_diff_gpu = compute_differences(data[f"pytorch_{op}"], data[f"gpu_{op}"])
    max_diff,avg_diff, min_diff = compute_differences(data[f"cpu_{op}"], data[f"gpu_{op}"])
    
    results.append([op, 
                    f"{max_diff_cpu:.5f} /{avg_diff_cpu:.5f} / {min_diff_cpu:.5f}", 
                    f"{max_diff_gpu:.5f} / {avg_diff_gpu:.5f} /{min_diff_gpu:.5f}",
                    f"{max_diff:.5f} /{avg_diff:.5f} / {min_diff:.5f}"])

# Create DataFrame
df = pd.DataFrame(results, columns=["Operation", "PyTorch vs CPU (Max/Avg/Min)", "PyTorch vs GPU (Max/Avg/Min)","CPU vs GPU (Max/Avg/Min)"])

# Print structured table
print(df.to_string(index=False))
