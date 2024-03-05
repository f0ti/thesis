The goal is to generate training data using headless server, to avoid using GUI interface. The input will be the LAS file and the output should be a set of rasterized images that contain per pixel information of x, y, z coordinates + RGB color format + other sensor information required.

To decide for the number of images to take from one scene we can take into consideration:
- Color distribution
- Number of points coverage, absolute or relative

For the rendering process, I looked into the Open3D python package and it provides an [example](https://github.com/isl-org/Open3D/blob/f5f672b4af1fc81e423c3c1b6215497f5a8816ea/examples/python/visualization/headless_rendering.py) on how to perform this. In order to make use of that feature, the flag ENABLE_HEADLESS_RENDERING needs to be set ON while building the package. The issue is that it is not possible to build the package with both ENABLE_HEADLESS_RENDERING and BUILD_GUI flags enabled, as issued [here](https://github.com/isl-org/Open3D/issues/2998). It is not a huge necessity, but the GUI might be useful to have as a testing environment. The solution is to have two conda environments for each.

There is also an alternative python package called pyrender, which is more lightweight and simpler than open3D, that offers a scene viewer and a headless server with GPU support.