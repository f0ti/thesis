import os
import numpy as np
import open3d as o3d
import laspy
from dataset.utils.point import Point

DATA_DIR = "../data/CoM_Point_Cloud_2018_LAS"

class PointCloudReader:
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir

    def read_pcd(self, filename):
        file_dir = os.path.join(self.data_dir, filename)
        pcd = o3d.io.read_point_cloud(file_dir)
        print(pcd)

        return pcd


class LasReader:
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir

    def read_las(self, filename):
        file_dir = os.path.join(self.data_dir, filename)
        
        with laspy.open(file_dir, mode='r') as fh:
            print('Points from Header:', fh.header.point_count)
            las = fh.read()
            print('Points from data:', len(las.points))

            return las


def main():
    filename = "Tile_+003_+005.las"
    lasr = LasReader(DATA_DIR)
    las = lasr.read_las(filename)


if __name__ == "__main__":
    main()
