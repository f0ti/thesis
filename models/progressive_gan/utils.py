import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch import Tensor

from losses import CycleGANLoss, WganGP, StandardGAN


def adjust_dynamic_range(
    data: Tensor,
    drange_in: Tuple[float, float] = (0.0, 1.0),
    drange_out: Tuple[float, float] = (-1.0, 1.0),
):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
            np.float32(drange_in[1]) - np.float32(drange_in[0])
        )
        bias = np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale
        data = data * scale + bias

    return torch.clamp(data, min=drange_out[0], max=drange_out[1])


def post_process_generated_images(imgs: Tensor) -> np.ndarray:
    imgs = adjust_dynamic_range(
        imgs, drange_in=(-1.0, 1.0), drange_out=(0.0, 1.0)
    )
    return (imgs * 255.0).detach().cpu().numpy().astype(np.uint8)


def post_process_coordinate_images(imgs: Tensor) -> np.ndarray:
    imgs = adjust_dynamic_range(
        imgs, drange_in=(-1.0, 1.0), drange_out=(0.0, 1.0)
    )
    return imgs.detach().cpu().numpy()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# noinspection PyPep8Naming
def str2GANLoss(v):
    if v.lower() == "wgan_gp":
        return WganGP()
    elif v.lower() == "cycle_gan":
        return CycleGANLoss()
    elif v.lower() == "standard_gan":
        return StandardGAN()
    else:
        raise argparse.ArgumentTypeError(
            "Unknown gan-loss function requested."
            f"Please consider contributing your GANLoss to: "
            f"{str(Path(losses.__file__).absolute())}"  # type: ignore
        )


def show_rgb(images, cols=4, figsize=(20, 20)):
    rows = len(images) // cols
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(axs.flat):
        img = images[i]
        img = img.transpose((1, 2, 0))
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_xyz(images, cols=4, figsize=(20, 20)):
    rows = len(images) // cols
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(axs.flat):
        # show only Z channel
        img = images[i]
        ax.imshow(img[2])
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_diff(ground_truth, predicted, cols=4, figsize=(20, 20)):
    rows = (len(ground_truth) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flat
    
    for i in range(len(ground_truth)):
        gt = ground_truth[i].transpose((1, 2, 0))
        pred = predicted[i].transpose((1, 2, 0))
        diff = np.abs(gt - pred)
        
        # If images are colored, average the differences across the channels to get a single-channel heatmap
        if diff.ndim == 3:
            diff = np.mean(diff, axis=2)
        ax = axs[i]
        sns.heatmap(diff, cmap='rainbow', cbar=False, xticklabels=False, yticklabels=False, ax=ax)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def print_weights(model):
    for name, param in model.named_parameters():
        print(name, param)
