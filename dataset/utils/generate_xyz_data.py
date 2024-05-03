import os
import click
from tqdm import tqdm
from tile import Tile

@click.command()
@click.argument("mode", type=click.Choice(["train", "test"]))
def generate_xyz_data(mode):

    assert mode in ['train', 'test']

    base_dir = f"../melbourne/tiles_data/{mode}"
    save_dir = f"../melbourne/xyz_data_experiment/{mode}"
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    paths = os.listdir(base_dir)

    for path in tqdm(paths):
        file_dir = os.path.join(base_dir, path)
        if os.stat(file_dir).st_size < 100:
            continue
        tile = Tile(file_dir)
        tile.save_xyz(save_dir)

if __name__ == "__main__":
    generate_xyz_data()
