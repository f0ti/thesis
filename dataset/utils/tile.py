import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from struct import unpack
from typing import List

from point import Point

QUIET = True


class Tile:
    def __init__(self, filename: str) -> None:
        self.filename: str = filename
        self.points = []
        self.read_points()

    @property
    def RGB(self) -> List:
        return [(point.R, point.G, point.B) for point in self.points]

    @property
    def XYZ(self) -> List:
        return [(point.X, point.Y, point.Z) for point in self.points]

    @property
    def I(self) -> List:
        return [point.intensity for point in self.points]

    @property
    def X(self) -> List:
        return [point.X for point in self.points]

    @property
    def Y(self) -> List:
        return [point.Y for point in self.points]

    @property
    def Z(self) -> List:
        return [point.Z for point in self.points]

    @property
    def pointIds(self) -> List:
        return [point.pointId for point in self.points]

    def read_points(self) -> None:
        with open(self.filename, "rb") as reader:
            # read header
            h_data: tuple = unpack("ff", reader.read(8))
            self.width = int(h_data[0])
            self.height = int(h_data[1])
            _ = int.from_bytes(reader.read(4), byteorder="little")
            self.nr_points = int(self.width * self.height)

            for _ in tqdm(range(self.nr_points), disable=QUIET):
                data = unpack("<dddBBBBI?", reader.read(33))

                x = data[0]
                y = data[1]
                z = data[2]
                r = data[3]
                g = data[4]
                b = data[5]
                intensity = data[6]
                point_id = data[7]
                isEmpty = data[8]

                point = Point(x, y, z, r, g, b, intensity, point_id, isEmpty)
                self.points.append(point)

    def show(self) -> None:
        plt.imshow(np.asarray(self.RGB, dtype=np.uint8).reshape(self.width, self.height, 3))
        plt.show()

    @staticmethod
    def _scale_xyz(xyz: np.ndarray) -> np.ndarray:
        xyz = np.clip(xyz, 0, None, dtype=np.float32)  #  set min=0
        
        # theoretically, we expect X and Y data values to be between 0-500 and Z between 0-50
        xyz[:, :, 0] /= 500
        xyz[:, :, 1] /= 500
        xyz[:, :, 2] /= 50

        return np.clip(xyz, 0, 1, dtype=np.float32)  # clip values to 0-1

    @staticmethod
    def _scale_z(z: np.ndarray) -> np.ndarray:
        z /= 50.0
        return np.clip(z, 0, 1, dtype=np.float32)

    @staticmethod
    def _scale_255(img: np.ndarray) -> np.ndarray:
        img /= 255.0
        return img

    def save_rgb(self, root) -> None:
        img = np.asarray(self.RGB, dtype=np.float32).reshape(
            self.width, self.height, 3
        )
        img = self._scale_255(img)
        img_name = self.filename.split("/")[-1]
        np.save(f"{root}/{img_name}.npy", img)

    def save_xyz(self, root) -> None:
        xyz = np.asarray(self.XYZ, dtype=np.float32).reshape(self.width, self.height, 3)
        xyz_name = self.filename.split("/")[-1]
        np.save(f"{root}/{xyz_name}.npy", xyz)

    def save_z(self, root) -> None:
        z = np.asarray(self.Z, dtype=np.float32).reshape(self.width, self.height, 1)
        z_name = self.filename.split("/")[-1]
        np.save(f"{root}/{z_name}.npy", z)

    def save_i(self, root) -> None:
        i = np.asarray(self.I, dtype=np.float32).reshape(self.width, self.height, 1)
        i = self._scale_255(i)
        i_name = self.filename.split("/")[-1]
        np.save(f"{root}/{i_name}.npy", i)

    def save_zi(self, root) -> None:
        z = np.asarray(self.Z, dtype=np.float32).reshape(self.width, self.height, 1)
        i = np.asarray(self.I, dtype=np.float32).reshape(self.width, self.height, 1)
        i = self._scale_255(i)
        z_i = np.concatenate((z, i), axis=2)
        z_i_name = self.filename.split("/")[-1]
        np.save(f"{root}/{z_i_name}.npy", z_i)

    def save_dtm(self, root) -> None:
        z_dtm = np.asarray(self.Z, dtype=np.float32).reshape(self.width, self.height, 1)
        dtm_name = self.filename.split("/")[-1]
        np.save(f"{root}/{dtm_name}.npy", z_dtm)
