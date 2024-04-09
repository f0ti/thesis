import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from struct import unpack
from point import Point

class Tile:
    
    def __init__(self, filename) -> None:
        self.filename = filename
        self.points = []
        self.read_points()

    @property
    def colors(self):
        return [(point.R, point.G, point.B) for point in self.points]

    @property
    def intensities(self):
        return [point.intensity for point in self.points]
    
    @property
    def xyz(self):
        return [(point.X, point.Y, point.Z) for point in self.points]

    @property
    def pointIds(self):
        return [point.pointId for point in self.points]

    def read_points(self):
        with open(self.filename, 'rb') as reader:
            # read header
            h_data = unpack('ff', reader.read(8))
            self.width = int(h_data[0])
            self.height = int(h_data[1])
            self.nr_points = int.from_bytes(reader.read(4), byteorder='little')

            for _ in tqdm(range(self.nr_points)):
                data = unpack('<dddHHHBI?', reader.read(36))

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

            print(f"Read {self.nr_points} points from {self.filename}")

    def show(self):
        # search if tile name exists in imgs folder
        img_name = self.filename.split('/')[-1]
        try:
            img = plt.imread(f"imgs/{img_name}.png")
        except FileNotFoundError:            
            img = np.asarray(self.colors, dtype=np.uint8).reshape(self.width, self.height, 3)
        
        plt.imshow(img)
        plt.show()

    def save(self):
        img = np.asarray(self.colors, dtype=np.uint8).reshape(self.width, self.height, 3)
        img_name = self.filename.split('/')[-1]
        plt.imsave(f"imgs/{img_name}.png", img)
        plt.show()
