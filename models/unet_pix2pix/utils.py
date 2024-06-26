import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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