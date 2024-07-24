import os
import sys
import numpy as np
from PIL import Image

class DataCheck:
    def __init__(self, dataset_path) -> None:
        self.dataset_path = dataset_path

    def check_RGB(self, debug=False):

        folder_path = os.path.join(self.dataset_path, "rgb_data")
        paths = sorted(os.listdir(folder_path))

        for path in paths:
            file_dir = os.path.join(folder_path, path)
            img = np.load(file_dir)
            if debug:
                print(f"Image path: {file_dir}")
                print(f"Image shape: {img.shape}")
                print(f"Image max: {img.max()}")
                print(f"Image min: {img.min()}")
                print(f"Image mean: {img.mean()}")
                print(f"Image std: {img.std()}")

            assert img.shape == (256, 256, 3), "Image shape is not (256, 256, 3)"

        print("RGB data check passed")

    def check_XYZ(self, debug=False):

        folder_path = os.path.join(self.dataset_path, "xyz_data")
        paths = sorted(os.listdir(folder_path))

        for path in paths:
            file_dir = os.path.join(folder_path, path)
            xyz = np.load(file_dir)
            if debug:
                print(f"XYZ path: {file_dir}")
                print(f"XYZ shape: {xyz.shape}")
                print(f"XYZ max: {xyz.max()}")
                print(f"XYZ min: {xyz.min()}")
                print(f"XYZ mean: {xyz.mean()}")
                print(f"XYZ std: {xyz.std()}")

            assert xyz.shape == (256, 256, 3), "XYZ shape is not (256, 256, 3)"
        
        print("XYZ data check passed")

    def check_Z(self, debug=False):

        folder_path = os.path.join(self.dataset_path, "z_data")
        paths = sorted(os.listdir(folder_path))

        for path in paths:
            file_dir = os.path.join(folder_path, path)
            z = np.load(file_dir)
            if debug:
                print(f"Z path: {file_dir}")
                print(f"Z shape: {z.shape}")
                print(f"Z max: {z.max()}")
                print(f"Z min: {z.min()}")
                print(f"Z mean: {z.mean()}")
                print(f"Z std: {z.std()}")

            assert z.shape == (256, 256, 1), "Z shape is not (256, 256, 1)"

        print("Z data check passed")


def main():
    dataset_name = sys.argv[1]
    datacheck = DataCheck(dataset_path=f"../{dataset_name}")
    datacheck.check_RGB()
    # datacheck.check_XYZ()
    datacheck.check_Z()
    

if __name__ == "__main__":
    main()
