## Keywords
cGAN, Differentiable Rendering, PointNet++, PatchGAN, Autoencoder, IEEE GRSS Data, Airborne LiDAR
## Novelty
uses a combination of cGAN (1 generator and 2 discriminators) and rendering
## Approach
The paper proposes a 3D point cloud colourisation schema (point2color) using a **cGAN** and **differentiable rendering**.

The airborne 3D point cloud evaluates the **colour point-by-point** and trains to minimise the loss function for each image by projecting it in 2D with a **differentiable renderer**, given the characteristics of spatially distributed objects.

**Steps**:
- given input patch with $N \times 3$ dimensions the **generator model** predicts the fake colour $N \times 3$ channels
- the **point discriminator** given $N \times 6\ (x, y, z, R, G, B)$ input patch, judges if the colour is real or fake
- the **image discriminator** given the $H \times W$ rendered image, judges if the image is real or fake
![Untitled](Untitled.png)
## Models
The usage of cGANs is trending in the general topic of colorization in Deep Learning. Before that CNN were used to learn the mappings from greyscale to colour images. This paper proposes an architecture consisting of 1 generator and 2 discriminators.

**Generator**
Since the correspondence between coordinates and colour in one-to-one an autoencoder is used as generator based on the **PointNet++** model. Skip connections are used, in order for the decoder to make use of the simpler features that were extracted in the earlier layers of the encoder. In the downsampling layer, the representative point is determined by furthest point sampling (FPS), and the information of the points around the representative point is collapsed.

**Point Discriminator**
Uses PointNet++-based PatchGAN that judges the fake colour of the generator and real colour of the ground truth.

**Image Discriminator**
Uses CNN-based PatchGAN to judge the colourised image by the generator and ground truth image. Based on pix2pix, a $70 \times 70$ PatchGAN architecture is used.
## Rendering
The second stage of the schema is the rendering of the image, which transforms the 3D structure output by the colourisation model to a 2D image. To do so, the camera parameters need to be known (position and direction of the ground truth image is taken), in order to input them to the virtual camera to generate the images.

**Differentiable rendering**
Since the neural network of the prediction model is trained on backpropagation of the difference in the output image, the error cannot be backpropagated unless the rendering is differentiable.

If the 3D structure is represented by a triangular mesh, the output image should change smoothly. For example, when a 3D structure represented by a triangular mesh is rendered using a rasterisation technique, the pixel values change discontinuously depending on whether the mesh is placed in the centre of the pixels. The output does not change smoothly in response to changes in the position of the mesh, so it is not differentiable.

In the case of mesh representation, the most common approach is to assume some initial shape and to reconstruct the 3D structure by deforming it (e.g., by first assuming a sphere as the mesh structure and then continuously deforming it into the desired object). The disadvantages of this approach are:

- difficult to reconstruct objects with different topologies (e.g., holes, multiple objects)
- a drawback of mesh representation is that it is not easy to handle data with complex structures consisting of vertices and faces in a deep learning framework
- a drawback of DRC, which uses voxel representation, is that the memory requirement increases with the cube of the resolution as the resolution increases

![https://raw.githubusercontent.com/ShichenLiu/SoftRas/master/data/media/demo/deform/sphere_to_car.gif](https://raw.githubusercontent.com/ShichenLiu/SoftRas/master/data/media/demo/deform/sphere_to_car.gif)
## Loss functions
**Pointwise loss** - emphasises the pointwise fidelity of the estimated colour by the generator, L1 distance of fake colour and real colour

**Pixelwise loss** - emphasises the rendered image fidelity of the estimated colour by the generator, L1 distance of fake image and real image

**Point GAN loss** - Wasserstein distance of fake colour and real colour

**Image GAN loss** - Wasserstein distance of fake image and real image
## Dataset
3D point clouds and aerial photos of an outdoor scene called the 2018 IEEE GRSS Data Fusion Contest

DFC2018 contains a wide variety of objects, such as houses, ground, trees, roads, and an American football field.

PDAL was used to create the real colour of each point from the aerial photo.

Due to its large size, it had to be split into small patches to be computed by the GPU. In this study, we split the airborne 3D point cloud with real colour into 30 $m^2$. The target for the actual loss calculation is the central 25 $m^2$ and the outside area was used to provide context information for colorization.
## Results
Training the model on the generator and two discriminators took approximately three days on a TSUBAME3.0 with one NVIDIA P100 GPU until 100 epochs.

**Quantitative Evaluation**
Mean Absolute Error of fake and real colour for each colour channel
MSE = 0.1

**Qualitative Evaluation**
more vivid colours

**Failure points**
the PointNet++ generator was ignoring small objects when colourised

Based on the [paper](https://www.notion.so/Scene-level-Point-Cloud-Colorization-with-Semantics-and-geometry-aware-Networks-c7377e044b3e49f1a340120b0117e41d?pvs=21):
- they only support the simple 3D object as input, which makes them unable to identify multiple objects and cluttered background
- they exhibit unsatisfactory artefacts, incoherent colours, or colourisation that homogeneously colour the entire scene regardless of differences between objects
## Comments
For the experimental setup **PyTorch** was used to implement the training process and image discriminator, **PyTorch Geometric** was used to implement PointNet++-based a generator and a point discriminator, and **PyTorch3D** was used to implement differentiable rendering for airborne 3D point clouds.
## Links

***Soft rasteriser*** - [https://github.com/ShichenLiu/SoftRas](https://github.com/ShichenLiu/SoftRas?tab=readme-ov-file)
***PointNet++*** - [https://github.com/charlesq34/pointnet2](https://github.com/charlesq34/pointnet2)
***pix2pix***  - [https://www.tensorflow.org/tutorials/generative/pix2pix](https://www.tensorflow.org/tutorials/generative/pix2pix)
***IEEE GRSS*** - [https://hyperspectral.ee.uh.edu/?page_id=1075](https://hyperspectral.ee.uh.edu/?page_id=1075)