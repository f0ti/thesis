import torch
import numpy as np
import matplotlib.pyplot as plt
from tile import Tile

torch.set_printoptions(sci_mode=False)

# ----------------
# Debug functions
# ----------------
def show_rgb(img):
    plt.imshow(img)
    plt.show()

def show_xyz(img):
    plt.imshow(img[:,:,2])
    plt.show()

input_path = "/home/foti/aerial/thesis/dataset/melbourne/xyz_data/test/Tile_+003_+005_0_3_ov.npy"
label_path = "/home/foti/aerial/thesis/dataset/melbourne/rgb_data/test/Tile_+003_+005_0_3_ov.npy"
input = np.load(input_path)
label = np.load(label_path)

# do some statistical analysis on the input
print("Input statistics")
print("Mean: ", np.mean(input))
print("Std: ", np.std(input))
print("Max: ", np.max(input))
print("Min: ", np.min(input))
print("Shape: ", input.shape)
print("Type: ", input.dtype)
print("Unique: ", np.unique(input))
print("Unique count: ", np.unique(input, return_counts=True))
