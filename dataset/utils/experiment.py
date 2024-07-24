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

dataset_path = "../melbourne-z-top/rgb_data"
for dir in os.listdir(dataset_path):
    img = np.load(os.path.join(dataset_path, dir))
    show_rgb(img)
    print(img.shape)
    break

# for dir in os.listdir("../melbourne-top/cluster_0"):
#     img = np.load(f"../melbourne-top/cluster_0/{dir}")
#     show_rgb(img)

# for dir in os.listdir("../melbourne-top/cluster_1"):
#     img = np.load(f"../melbourne-top/cluster_1/{dir}")
#     show_rgb(img)

# for dir in os.listdir("../melbourne-top/xyz_data"):
#     img = np.load(f"../melbourne-top/xyz_data/{dir}")
#     print(img.min(), img.max())
#     show_rgb(img)
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

