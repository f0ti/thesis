import numpy as np
from tile import Tile

filename = "Tile_+020_+013_4_0_ov.npy"
file_dir = f"../melbourne/xyz_data_experiment/test/{filename}"
tile_dir = "/home/foti/aerial/thesis/dataset/melbourne/LasFiles_30-04-2024/LasFiles/Tile_+005_+017/Tile_+005_+017/Tile_+005_+017_0_0_ov"

tile = Tile(tile_dir)
# tile_xyz = np.load(file_dir)
# print(tile.xyz)
# print(tile_xyz)
