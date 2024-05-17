import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from struct import unpack
from point import Point
from typing import List

QUIET = True

class Tile:
    
    def __init__(self, filename) -> None:
        self.filename = filename
        self.points = []
        self.read_points()

    @property
    def colors(self) -> List:
        return [(point.R, point.G, point.B) for point in self.points]

    @property
    def intensities(self) -> List:
        return [point.intensity for point in self.points]
    
    @property
    def xyz(self) -> List:
        return [(point.X, point.Y, point.Z) for point in self.points]

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
        with open(self.filename, 'rb') as reader:
            # read header
            h_data: tuple = unpack('ff', reader.read(8))
            self.width = int(h_data[0])
            self.height = int(h_data[1])
            self.nr_points = int.from_bytes(reader.read(4), byteorder='little')

            for _ in tqdm(range(self.nr_points), disable=QUIET):
                data = unpack('<dddBBBBI?', reader.read(33))

                x = data[0]
                y = data[1]
                z = data[2]
                r = data[3]
                g = data[4]
                b = data[5]
                intensity = data[6]
                point_id = data[7]
                bool = data[8]

                point = Point(x, y, z, r, g, b, intensity, point_id, bool)
                self.points.append(point)

    def show(self) -> None:
        # search if tile name exists in imgs folder
        img_name = self.filename.split('/')[-1]
        try:
            img = plt.imread(f"imgs/{img_name}.png")
        except FileNotFoundError:
            img = np.asarray(self.colors, dtype=np.uint8).reshape(self.width, self.height, 3)
        
        plt.imshow(img)
        plt.show()

    def save_rgb(self, root) -> None:
        img = np.asarray(self.colors, dtype=np.uint8).reshape(self.width, self.height, 3)
        img_name = self.filename.split('/')[-1]
        plt.imsave(f"{root}/{img_name}.png", img)

    def save_dtm(self, root) -> None:
        Z = np.asarray(self.Z, dtype=np.float64).reshape(self.width, self.height, 1)
        Z = np.clip(Z, 0, 50, dtype=np.float16)
        dtm_name = self.filename.split('/')[-1]
        np.save(f"{root}/{dtm_name}.npy", Z)

    def save_xyz(self, root) -> None:
        xyz = np.asarray(self.xyz, dtype=np.double).reshape(self.width, self.height, 3)
        xyz_name = self.filename.split('/')[-1]
        np.save(f"{root}/{xyz_name}.npy", xyz)
