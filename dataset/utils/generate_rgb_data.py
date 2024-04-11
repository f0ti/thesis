import os
import click
from tile import Tile

@click.command()
@click.argument("mode", type=click.Choice(["train", "test"]))
def generate_rgb_data(mode):

    assert mode in ['train', 'test']

    base_dir = f"../melbourne/tiles_data/{mode}"
    save_dir = f"../melbourne/rgb_data/{mode}"
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    paths = os.listdir(base_dir)

    for path in paths:
        tile = Tile(os.path.join(base_dir, path))
        tile.save_rgb(save_dir)

if __name__ == "__main__":
    generate_rgb_data()
