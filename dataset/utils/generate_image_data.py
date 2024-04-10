from fileinput import filename
from tile import Tile
import os

def generate_image_data(mode="train"):

    assert mode in ['train', 'test']

    base_dir = f"../tiles_data/melbourne/{mode}"
    save_dir = f"../image_data/melbourne/{mode}"
    paths = os.listdir(base_dir)

    for path in paths:
        tile = Tile(os.path.join(base_dir, path))
        tile.save(save_dir)

generate_image_data('test')
