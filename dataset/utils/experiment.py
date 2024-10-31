import matplotlib.pyplot as plt

from tile import Tile

# ----------------
# Debug functions
# ----------------
def show_rgb(img):
    plt.imshow(img)
    plt.show()

def show_xyz(img):
    plt.imshow(img, cmap="gray")
    plt.show()

def show_z(img):
    plt.imshow(img)
    plt.show()

# read tile
tile_id = "589551_2_2_ov"
tile_path = f"/home/foti/aerial/thesis/dataset/estonia_tiles/589551/{tile_id}"
tile = Tile(tile_path)
tile.show()

col_path = f"/home/foti/aerial/thesis/colorized_output/{tile_id}_colorized.bin"
col_tile = Tile(col_path)
col_tile.show()
