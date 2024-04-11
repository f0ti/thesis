import os
import click
from tile import Tile

@click.command()
@click.argument("mode", type=click.Choice(["train", "test"]))
def generate_xyz_data(mode):

    assert mode in ['train', 'test']

    base_dir = f"../melbourne/tiles_data/{mode}"
    save_dir = f"../melbourne/xyz_data/{mode}"
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    paths = os.listdir(base_dir)

    for path in paths:
        tile = Tile(os.path.join(base_dir, path))
        tile.save_xyz(save_dir)

if __name__ == "__main__":
    generate_xyz_data()
