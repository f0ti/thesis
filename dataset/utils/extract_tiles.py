import os
import sys
import fire

from tile import Tile

class DataExtractor():
    def __init__(self, dataset: str, data_type: list) -> None:
        assert dataset in ["melbourne-top", "melbourne-z-top", "estonia", "estonia-staging"]
        self.dataset = dataset
        if self.dataset in ["estonia", "estonia-staging"]:
            self.tile_bytesize = 2228236
        else:
            self.tile_bytesize = 2162700

        assert all([data in ["z", "i", "rgb", "zi", "xyz"] for data in data_type]), "Invalid data type"
        self.data_type = data_type

        self.base_dir = f"../{self.dataset}/tiles_data"

        for data in data_type:
            os.makedirs(f"../{self.dataset}/{data}_data", exist_ok=True)

    def _correct_bytesize(self, file_dir: str):
        if os.stat(file_dir).st_size != self.tile_bytesize:
            return False
        return True

    def generate(self) -> None:
        data_paths = os.listdir(self.base_dir)
        print(f"Extracting {self.data_type} data from {len(data_paths)} tiles")
        for i, path in enumerate(data_paths):
            sys.stdout.write(f"\rProcessing [{i+1}/{len(data_paths)}] {path}")
            
            file_dir = os.path.join(self.base_dir, path)
            if not self._correct_bytesize(file_dir):
                continue
            
            tile = Tile(file_dir)
            
            if "rgb" in self.data_type:
                tile.save_rgb(f"../{self.dataset}/rgb_data")
            if "z" in self.data_type:
                tile.save_z(f"../{self.dataset}/z_data")
            if "i" in self.data_type:
                tile.save_i(f"../{self.dataset}/i_data")
            if "zi" in self.data_type:
                tile.save_zi(f"../{self.dataset}/zi_data")
            if "xyz" in self.data_type:
                tile.save_xyz(f"../{self.dataset}/xyz_data")

if __name__ == "__main__":
    fire.Fire(DataExtractor)
