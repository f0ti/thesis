import random
import time
import datetime
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable

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
    plt.show()


def show_xyz(images, cols=4, figsize=(20, 20)):
    rows = len(images) // cols
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(axs.flat):
        # show only Z channel
        img = images[i]
        ax.imshow(img[2])
        ax.axis('off')
    plt.show()


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
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)