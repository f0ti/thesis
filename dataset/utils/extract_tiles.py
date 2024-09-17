import os
import sys
from tile import Tile

# change for different dataset
class Extractor():
    def __init__(self, dataset_name, mode: str="z") -> None:
        assert dataset_name in ["melbourne-top", "melbourne-z-top", "estonia"]
        self.dataset_name = dataset_name
        if self.dataset_name == "estonia":
            self.tile_bytesize = 2228236
        else:
            self.tile_bytesize = 2162700

        assert mode in ["z", "xyz", "zi", "i"]
        self.mode = mode

        self.base_dir = f"../{self.dataset_name}/tiles_data"
        self.rgb_dir  = f"../{self.dataset_name}/rgb_data"
        self.coo_dir  = f"../{self.dataset_name}/{self.mode}_data"

        self.num_tiles = len(os.listdir(self.base_dir))

        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.coo_dir, exist_ok=True)

    def generate_data(self) -> None:
        for i, path in enumerate(os.listdir(self.base_dir)):
            sys.stdout.write(f"\rProcessing [{i+1}/{self.num_tiles}] {path}")
            file_dir = os.path.join(self.base_dir, path)
            if os.stat(file_dir).st_size != self.tile_bytesize:
                continue
            
            tile = Tile(file_dir)
            tile.save_rgb(self.rgb_dir)

            if self.mode == "z":
                tile.save_z(self.coo_dir)
            elif self.mode == "i":
                tile.save_i(self.coo_dir)
            elif self.mode == "zi":
                tile.save_zi(self.coo_dir)
            elif self.mode == "xyz":
                tile.save_xyz(self.coo_dir)
            else:
                raise ValueError("Invalid type")

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    mode = sys.argv[2]
    extractor = Extractor(dataset_name, mode)
    extractor.generate_data()
