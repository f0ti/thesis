import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from data import RGBTileDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from torch import Tensor

# Plot the images
def show_images(images, cols=4, figsize=(20, 20)):
    rows = len(images) // cols
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(axs.flat):
        # Transpose the image from (3, 256, 256) to (256, 256, 3)
        img = images[i].astype(np.uint8)
        ax.imshow(img)
        ax.axis('off')
    plt.show()

train_dataset = RGBTileDataset(dataset="melbourne", image_set="train")
train_dataloader = DataLoader(train_dataset, batch_size=16)
images = next(iter(train_dataloader))[1]

images = images.permute(0, 2, 3, 1).numpy()
images = images * 255

seq = iaa.Sequential([
    iaa.Resize(0.5)  # resize images to 50%
    # iaa.Rotate([-180, -90, 90, 180]),  # rotate by -45 to 45 degrees
]) # apply augmenters in random order

images_aug = seq(images=images)

show_images(images_aug) # Show the augmented images
