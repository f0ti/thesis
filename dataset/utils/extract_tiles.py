import os
import sys
from tqdm import tqdm
from tile import Tile

# change for different dataset
TILE_BYTESIZE = 2162700


def generate_rgb_data(dataset_name):

    assert dataset_name in ["melbourne-top", "melbourne-z-top"]

    base_dir = f"../{dataset_name}/tiles_data"
    save_dir = f"../{dataset_name}/rgb_data"

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    paths = os.listdir(base_dir)

    print("Generating RGB data")
    for path in tqdm(paths):
        file_dir = os.path.join(base_dir, path)
        if os.stat(file_dir).st_size != TILE_BYTESIZE:
            continue
        tile = Tile(file_dir)
        tile.save_rgb(save_dir)


def generate_xyz_data(dataset_name):

    assert dataset_name in ["melbourne-top", "melbourne-z-top"]

    base_dir = f"../{dataset_name}/tiles_data"
    save_dir = f"../{dataset_name}/xyz_data"

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    paths = os.listdir(base_dir)

    print("Generating XYZ data")
    for path in tqdm(paths):
        file_dir = os.path.join(base_dir, path)
        if os.stat(file_dir).st_size != TILE_BYTESIZE:
            continue
        tile = Tile(file_dir)
        tile.save_xyz(save_dir)


def generate_z_data(dataset_name):

    assert dataset_name in ["melbourne-top", "melbourne-z-top"]

    base_dir = f"../{dataset_name}/tiles_data"
    save_dir = f"../{dataset_name}/z_data"

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    paths = os.listdir(base_dir)

    print("Generating Z data")
    for path in tqdm(paths):
        file_dir = os.path.join(base_dir, path)
        if os.stat(file_dir).st_size != TILE_BYTESIZE:
            continue
        tile = Tile(file_dir)
        tile.save_z(save_dir)

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    generate_rgb_data(dataset_name)
    generate_xyz_data(dataset_name)
    # generate_z_data(dataset_name)
