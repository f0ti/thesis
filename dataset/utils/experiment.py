import os
import cv2
from matplotlib.pylab import f
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

# for dir in os.listdir("../melbourne-top/train/rgb_data"):
#     img = np.load(f"../melbourne-top/train/rgb_data/{dir}")
#     show_rgb(img)
#     break

for dir in os.listdir("../melbourne-top/cluster_0"):
    img = np.load(f"../melbourne-top/cluster_0/{dir}")
    show_rgb(img)

for dir in os.listdir("../melbourne-top/cluster_1"):
    img = np.load(f"../melbourne-top/cluster_1/{dir}")
    show_rgb(img)

# for dir in os.listdir("../melbourne-top/xyz_data"):
#     img = np.load(f"../melbourne-top/xyz_data/{dir}")
#     print(img.min(), img.max())
#     show_rgb(img)
#     break
# tile = Tile(img)