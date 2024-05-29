import numpy as np
from tile import Tile

input_path = "/home/foti/aerial/thesis/dataset/melbourne/xyz_data/test/Tile_+003_+005_0_3_ov.npy"
label_path = "/home/foti/aerial/thesis/dataset/melbourne/rgb_data/test/Tile_+003_+005_0_3_ov.npy"
input = np.load(input_path)
label = np.load(label_path)

print(input.shape)
print(label.shape)
print(input.dtype)
print(label.dtype)
