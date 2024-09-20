import torch
import random
import numpy as np
import seaborn as sns
import torch.nn.init as init
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from typing import Tuple
from torch import Tensor
from torch.optim import lr_scheduler


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


def get_scheduler(optimizer, n_epochs, n_epochs_decay):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - n_epochs) / float(n_epochs_decay + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler


def init_weights(m, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    
    Parameters:
        m (nn.Module): network to initialize
        init_type (str): initialization method: normal | xavier | kaiming
        init_gain (float): scaling factor for normal, xavier and kaiming.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    m.apply(init_func)

# ----------------
# Debug functions
# ----------------

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

def show_z(images, cols=4, figsize=(20, 20)):
    images = images.cpu().detach().numpy()
    rows = len(images) // cols
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(axs.flat):
        # show only Z channel
        img = images[i]
        img = np.transpose(img, (1, 2, 0))
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def show_grid(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

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

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


def print_model_parameters(model):
    for p in model.named_parameters():
        print(p)
