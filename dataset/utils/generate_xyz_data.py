import os
import click
from tqdm import tqdm
from tile import Tile

TILE_BYTESIZE = 2162700

@click.command()
@click.argument("mode", type=click.Choice(["train", "test"]))
def generate_xyz_data(mode):

    assert mode in ['train', 'test']

    base_dir = f"../melbourne/tiles_data/{mode}"
    save_dir = f"../melbourne/xyz_data/{mode}"
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    paths = os.listdir(base_dir)

    print(f"Generating XYZ data {mode}")
    for path in tqdm(paths):
        file_dir = os.path.join(base_dir, path)
        if os.stat(file_dir).st_size != TILE_BYTESIZE:
            continue
        tile = Tile(file_dir)
        tile.save_xyz(save_dir)

if __name__ == "__main__":
    generate_xyz_data()
