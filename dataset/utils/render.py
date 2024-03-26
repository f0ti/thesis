import pylas
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.cbook import get_sample_data
from matplotlib.colors import LightSource
from rasterio.crs import CRS
from rasterio.transform import Affine

def read_las(filename):
    las = pylas.read(filename)
    return las

def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = np.nanmin(array), np.nanmax(array)
    return ((array - array_min)/(array_max - array_min))

def info(red, green, blue):
    print('datatype:', red.dtype, '-', 'min:', np.nanmin(red), '-', 'mean:', np.nanmean(red), '-', 'max:', np.nanmax(red))
    print('datatype:', green.dtype, '-', 'min:', np.nanmin(green), '-', 'mean:', np.nanmean(green), '-', 'max:', np.nanmax(green))
    print('datatype:', blue.dtype, '-', 'min:', np.nanmin(blue), '-', 'mean:', np.nanmean(blue), '-', 'max:', np.nanmax(blue))

filename = "../data/melbourne/Tile_+003_+005.las"

input_las = read_las(filename)
print(input_las.header)
print(input_las.point_format.dimension_names)

VLRList = input_las.vlrs
print(VLRList)
print(VLRList.get('GeoAsciiParamsVlr')[0])

points = list(zip(input_las.x, input_las.y))
print(len(points))

# Assign band variable
red = input_las.red
green = input_las.green
blue = input_las.blue

redn = normalize(red)
greenn = normalize(green)
bluen = normalize(blue)

info(red, green, blue)

resolution = 1

# Create coord ranges over the desired raster extension
xRange = np.arange(input_las.x.min(), input_las.x.max() + resolution, resolution)
yRange = np.arange(input_las.y.min(), input_las.y.max() + resolution, resolution)

gridX, gridY = np.meshgrid(xRange, yRange)

#interpolate the data
Red = griddata(points, redn, (gridX, gridY), method='linear')
Green = griddata(points, greenn, (gridX, gridY), method='linear')
Blue = griddata(points, bluen, (gridX, gridY), method='linear')

# Chekck the shape
print('shape of red band', Red.shape)
print('shape of green band', Green.shape)
print('shape of blue band', Blue.shape)

# Initialize subplots
fig, axs = plt.subplots(3, 1, figsize=(30, 17), sharey=True)

# Plot Red, Green and Blue (rgb)
axs[0].imshow(Red, cmap='Reds')
axs[1].imshow(Green, cmap='Greens')
axs[2].imshow(Blue, cmap='Blues')

# Add titles
axs[0].set_title("Red")
axs[1].set_title("Green")
axs[2].set_title("Blue")

# Show plot
plt.show()

# Create figure
fig = plt.figure(figsize=[20, 5])

# Create RGB natural color composite
RGB = np.dstack((Red, Green, Blue))

# Let's see how our color composite looks like
plt.imshow(RGB)

# Customize plot
plt.title('True color image', fontweight='bold')
plt.xlabel('width')
plt.ylabel('height')

# Show plot
plt.show()