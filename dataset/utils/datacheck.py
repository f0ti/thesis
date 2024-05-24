import os
import numpy as np
from PIL import Image

class DataCheck:
    def __init__(self, dataset_path) -> None:
        self.dataset_path = dataset_path

    def check_RGB(self, folder="test"):

        folder_path = os.path.join(self.dataset_path, "rgb_data", folder)
        paths = sorted(os.listdir(folder_path))

        for path in paths:
            file_dir = os.path.join(folder_path, path)
            img = Image.open(file_dir)
            img = np.array(img)
            print(f"Image path: {file_dir}")
            print(f"Image shape: {img.shape}")
            print(f"Image max: {img.max()}")
            print(f"Image min: {img.min()}")
            print(f"Image mean: {img.mean()}")
            print(f"Image std: {img.std()}")
            print(img)

            assert img.shape == (256, 256, 3), "Image shape is not (256, 256, 3)"

    # test dataset has correct values of XYZ

    def check_XYZ(self, folder="test"):

        folder_path = os.path.join(self.dataset_path, "xyz_data", folder)
        paths = sorted(os.listdir(folder_path))
        
        for path in paths:
            file_dir = os.path.join(folder_path, path)
            xyz = np.load(file_dir)
            print(f"XYZ path: {file_dir}")
            print(f"XYZ shape: {xyz.shape}")
            print(f"XYZ max: {xyz.max()}")
            print(f"XYZ min: {xyz.min()}")
            print(f"XYZ mean: {xyz.mean()}")
            print(f"XYZ std: {xyz.std()}")
            print(xyz)

            assert xyz.shape == (256, 256, 3), "XYZ shape is not (256, 256, 3)"


if __name__ == "__main__":
    datacheck = DataCheck(dataset_path="../melbourne")
    datacheck.check_RGB()
    datacheck.check_XYZ()
