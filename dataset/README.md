# Dataset

This module contains all scripts needed for generating the dataset.

## Melbourne

For the ![melbourne dataset](https://discover.data.vic.gov.au/dataset/city-of-melbourne-3d-point-cloud-2018) we have the following structure:

- ```raw_data``` - extracted .las data and tiles from zip
- ```las_data``` - contains the .las data
- ```rgb_data``` - contains only the RGB data extracted from tiles saved as numpy tensors
- ```xyz_data``` - contains only the XYZ data extracted from tiles saved as numpy tensors
- ```tiles_data``` - contains the tiles extracted from .las data in binary

running ```utils/extract_tiles.py``` will create ```train``` and ```test``` directories with each containing ```rgb_data``` and ```xyz_data```