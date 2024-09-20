import argparse
import imageio
import os
from natsort import natsorted
from requests import get

parser = argparse.ArgumentParser()
parser.add_argument("run_name")
opt = parser.parse_args()

input_dir = f"saved_images/{opt.run_name}"
output_dir = f"saved_gifs/{opt.run_name}_gifs"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# use this script to save each image as a gif
def save_each():
    # Get a sorted list of all image files in the input directory
    image_files = natsorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    # print(image_files)
    # Number of different images (n), determined from the pattern (e.g., _0, _1, _2, etc.)
    n = 5  # As per the provided example, n=5 (real_A_0 to real_A_4)

    # Group images by their last number (i.e., img_num)
    for img_num in range(n):
        images = []
        # Iterate over every n-th image that corresponds to the current img_num
        for i in range(img_num, len(image_files), n):
            image_path = os.path.join(input_dir, image_files[i])
            images.append(imageio.imread(image_path))

        # Save GIF for this group of images
        gif_path = os.path.join(output_dir, f"real_A_img_num_{img_num}.gif")
        imageio.mimsave(gif_path, images, duration=0.5)  # Adjust duration if needed

        print(f"Saved {gif_path}")

# use this script when the image contains them all in grid
def save_group(get_every=1):
    # Get a sorted list of all image files in the input directory
    image_files = natsorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    print(f"Found {len(image_files)} images")
    image_files = image_files[::get_every]
    print(f"Reduced to {len(image_files)} images")

    images = []
    for img in image_files:
        image_path = os.path.join(input_dir, img)
        images.append(imageio.v2.imread(image_path))

    gif_path = os.path.join(output_dir, f"progression_{len(images)}.gif")
    imageio.mimsave(gif_path, images, duration=0.5)  # Adjust duration if needed
    print(f"open {gif_path}")

save_group(get_every=5)