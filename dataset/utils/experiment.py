import os
import cv2
from matplotlib.pylab import f
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
    plt.imshow(img[:, :, 2])
    plt.show()

filename = "Tile_+014_+014_2_3.npy"
dir = "../melbourne-top/rgb_data/%s" % filename
img = np.load(dir)
show_rgb(img)

# dataset_path = "../melbourne-z-top/rgb_data"
# for dir in os.listdir(dataset_path):
#     img = np.load(os.path.join(dataset_path, dir))
#     show_rgb(img)
#     print(img.shape)
#     break

# for dir in os.listdir("../melbourne-top/cluster_0"):
#     img = np.load(f"../melbourne-top/cluster_0/{dir}")
#     show_rgb(img)

# for dir in os.listdir("../melbourne-top/cluster_1"):
#     img = np.load(f"../melbourne-top/cluster_1/{dir}")
#     show_rgb(img)

# for dir in os.listdir("../melbourne-top/xyz_data"):
#     print(dir)
#     img = np.load(f"../melbourne-top/xyz_data/{dir}")
#     print(img[0])
#     print(img.min(), img.max())
#     # show_xyz(img)
#     break
# tile = Tile(img)

# cache = 50
# path_cluster_0 = "../melbourne-top/cluster_0"
# path_cluster_1 = "../melbourne-top/cluster_1"
# cluster_0 = []
# cluster_1 = []
# for dir in os.listdir(path_cluster_0)[:cache]:
#     img = np.load(f"{path_cluster_0}/{dir}")
#     cluster_0.append(img)
# for dir in os.listdir(path_cluster_1)[:cache]:
#     img = np.load(f"{path_cluster_1}/{dir}")
#     cluster_1.append(img)

# # shuffle lists
# np.random.shuffle(cluster_0)
# np.random.shuffle(cluster_1)

# display_n = 5
# fig, axs = plt.subplots(display_n, display_n, figsize=(20, 20))
# for i in range(display_n):
#     for j in range(display_n):
#         axs[i, j].imshow(cluster_0[i * display_n + j])
#         axs[i, j].axis("off")
# plt.tight_layout()
# plt.show()

# fig, axs = plt.subplots(display_n, display_n, figsize=(20, 20))
# for i in range(display_n):
#     for j in range(display_n):
#         axs[i, j].imshow(cluster_1[i * display_n + j])
#         axs[i, j].axis("off")
# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Assuming 'xyz_data' is a 3D numpy array with shape (height, width, 3)
# # where xyz_data[:, :, 2] contains the Z values

# sample_filename = "../melbourne-z-top/test/z_data/Tile_+003_+005_4_2.npy"
# Z = np.load(sample_filename)

# print(Z.shape)

# # Normalize Z values
# Z_min = np.min(Z)
# Z_max = np.max(Z)
# Z_normalized = (Z - Z_min) / (Z_max - Z_min)

# print(Z_normalized.shape)

# # Map normalized Z values to colors using a colormap
# # 'jet' goes from blue (low values) to red (high values)
# colormap = plt.cm.jet
# elevation_map = colormap(Z_normalized)
# elevation_map = np.squeeze(elevation_map)

# print(elevation_map.shape)

# # Display the elevation map
# plt.imshow(elevation_map)
# plt.colorbar(label='Elevation')
# plt.show()
