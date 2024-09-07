import os
import sys
import numpy as np
from PIL import Image

class DataCheck:
    def __init__(self, dataset_path) -> None:
        self.dataset_path = dataset_path

    def check_RGB(self, data_mode: str, debug=False):

        folder_path = os.path.join(self.dataset_path, f"{data_mode}/rgb_data")
        paths = sorted(os.listdir(folder_path))

        for path in paths:
            file_dir = os.path.join(folder_path, path)
            img = np.load(file_dir)
            if debug:
                sys.stdout.write(f"Image path: {file_dir}")
                sys.stdout.write(f"Image shape: {img.shape}")
                sys.stdout.write(f"Image max: {img.max()}")
                sys.stdout.write(f"Image min: {img.min()}")
                sys.stdout.write(f"Image mean: {img.mean()}")
                sys.stdout.write(f"Image std: {img.std()}")

            assert img.shape == (256, 256, 3), "Image shape is not (256, 256, 3)"

        sys.stdout.write("RGB data check passed")

    def check_XYZ(self, data_mode: str, debug=False):

        folder_path = os.path.join(self.dataset_path, f"{data_mode}/rgb_data")
        paths = sorted(os.listdir(folder_path))

        for path in paths:
            file_dir = os.path.join(folder_path, path)
            xyz = np.load(file_dir)
            if debug:
                sys.stdout.write(f"XYZ path: {file_dir}")
                sys.stdout.write(f"XYZ shape: {xyz.shape}")
                sys.stdout.write(f"XYZ max: {xyz.max()}")
                sys.stdout.write(f"XYZ min: {xyz.min()}")
                sys.stdout.write(f"XYZ mean: {xyz.mean()}")
                sys.stdout.write(f"XYZ std: {xyz.std()}")

            assert xyz.shape == (256, 256, 3), "XYZ shape is not (256, 256, 3)"
        
        sys.stdout.write("XYZ data check passed")

    def check_Z(self, data_mode: str, debug=False):

        folder_path = os.path.join(self.dataset_path, f"{data_mode}/z_data")
        paths = sorted(os.listdir(folder_path))

        for path in paths:
            file_dir = os.path.join(folder_path, path)
            z = np.load(file_dir)
            if debug:
                # sys.stdout.write(f"\r Z path: {file_dir}")
                # sys.stdout.write(f"\r Z shape: {z.shape}")
                sys.stdout.write(f"\r Z max: {z.max()} Z min: {z.min()}")
                # sys.stdout.write(f"\r Z mean: {z.mean()}")
                # sys.stdout.write(f"\r Z std: {z.std()}")

            assert z.shape == (256, 256, 1), "Z shape is not (256, 256, 1)"

        sys.stdout.write("Z data check passed")


def main():
    dataset_name = sys.argv[1]
    datacheck = DataCheck(dataset_path=f"../{dataset_name}")
    # datacheck.check_RGB("train")
    # datacheck.check_RGB("test")
    # datacheck.check_XYZ()
    datacheck.check_Z("train", debug=True)
    datacheck.check_Z("test",  debug=True)
    

if __name__ == "__main__":
    main()
