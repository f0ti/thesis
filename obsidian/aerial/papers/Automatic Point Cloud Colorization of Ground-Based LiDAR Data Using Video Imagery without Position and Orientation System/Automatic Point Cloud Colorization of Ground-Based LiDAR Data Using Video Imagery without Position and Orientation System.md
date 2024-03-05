Created by: Foti
Created time: February 21, 2024 10:33 AM
Tags: 3D Projection, Outdoor Scenes
## Keywords
camera calibration, registration, normalised Zernike moments, corresponding point matching, essential matrix, relative orientation, absolute orientation, point cloud colouring
## Approach
In this paper, we propose a ground mobile measurement system only composed of a LiDAR and a GoPro camera, providing a more convenient and reliable way to automatically obtain 3D point cloud data with spectral information. The automatic point cloud colouring based on video images mainly includes four aspects

1. Establishing models for radial distortion and tangential distortion to correct video images.
2. Establishing a registration method based on normalised Zernike moments to obtain the exterior orientation elements.
3. Establishing relative orientation based on essential matrix decomposition and nonlinear optimisation. This involves uniformly using the speeded-up robust features algorithm with distance restriction and RANSAC to select corresponding points.
4. A point cloud colouring method based on Gaussian distribution with central region restriction is adopted. Only pixels within the central region are considered valid for colouring. Then, the point cloud is coloured based on the mean of the Gaussian distribution of the colour set. In the coloured point cloud, the textures of the buildings are clear, and targets such as windows, grass, trees, and vehicles can be clearly distinguished.
## Results
Overall, the result meets the accuracy requirements of applications such as tunnel detection, street-view modelling and 3D urban modelling.