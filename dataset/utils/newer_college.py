import os
import numpy as np
import open3d as o3d
from point import Point

DATA_DIR = "../data/newer_college/03_new_college_prior_map"


class MeshReader:
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir

    def read_mesh(self, filename):
        file_dir = os.path.join(self.data_dir, filename)
        mesh = o3d.io.read_triangle_mesh(file_dir)
        print(mesh)

        return mesh


class PointCloudReader:
    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir

    def read_pcd(self, filename):
        file_dir = os.path.join(self.data_dir, filename)
        pcd = o3d.io.read_point_cloud(file_dir)
        print(pcd)

        return pcd


class Converter:
    def __init__(self) -> None:
        pass

    def pcd_to_np(self, pcd):
        pcd_np = np.asarray(pcd.points)
        return pcd_np

    def pcd_to_points(self, pcd):
        pcd_np = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors)
        pcd_points = []
        for i in range(len(pcd_np)):
            p = Point(
                pcd_np[i][0],
                pcd_np[i][1],
                pcd_np[i][2],
                pcd_colors[i][0],
                pcd_colors[i][1],
                pcd_colors[i][2],
            )
            pcd_points.append(p)
        return pcd_points

    def np_to_pcd(self, pcd_np):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        return pcd

    def points_to_pcd(self, pcd_points):
        pcd = o3d.geometry.PointCloud()
        pcd_np = []
        pcd_colors = []
        for p in pcd_points:
            pcd_np.append([p.x, p.y, p.z])
            pcd_colors.append([p.r, p.g, p.b])
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
        return pcd


def main():
    filename = "new-college-29-01-2020-5cm-resolution.ply"
    pcdr = PointCloudReader(DATA_DIR)
    pcd = pcdr.read_pcd(filename)

    print(pcd.colors)

    # o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
