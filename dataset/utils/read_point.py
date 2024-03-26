from struct import unpack
from point import Point

points = []

with open('5Points', 'rb') as reader:
    
    # read header
    nr_points = int.from_bytes(reader.read(8), byteorder='little')
    
    for i in range(nr_points):
        data = unpack('<fffHHHH?', reader.read(21))

        x = data[0]
        y = data[1]
        z = data[2]
        r = data[3]
        g = data[4]
        b = data[5]
        intensity = data[6]
        bool = data[7]

        print(x, y, z, r, g, b, intensity, bool)

        point = Point(x, y, z, r, g, b, intensity, bool)
        points.append(point)

print(points)
