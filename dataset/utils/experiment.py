import os
import cv2
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
