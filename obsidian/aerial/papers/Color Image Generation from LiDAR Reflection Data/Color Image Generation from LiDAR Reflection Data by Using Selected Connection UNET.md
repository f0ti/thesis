Created by: Foti
Created time: February 21, 2024 12:53 PM
Tags: Deep Learning, Outdoor Scenes
## Keywords
artificial intelligence, heterogeneous transfer method, image generation, LiDAR sensor, LiDAR imaging, learning systems, selected-connection network, sparse input data
## Novelty
They propose a new methodology for the selection of the connections in UNET, taking into consideration the sparseness differences and similarities between the encoder and decoder
## Approach
The paper proposes using selected connection UNET (SC-UNET) architecture, which has been previously used for detecting and segmenting concealed objects in THz images for military purposes, as an extension of UNET architecture.

The input point clouds are extremely sparse from the output image. This difference in sparseness causes differences in the feature maps of the encoder and decoder, which can be identified in the levels of the network structure. The authors use the ED-FCN network to empirically analyse the similarities between encoder and decoder, and use these analyses for selecting connections in SC-UNET.
## Models
**ED-FCN**

3D LiDAR point clouds are converted into a 2D LiDAR reflection-intensity image using a projection matrix. The reflection image has the same spatial resolution as the RGB colour image to be generated.

The input reflection image is very sparse - 94.72%. The sparseness of the feature map is decreased as the data are forwarded through the contracting path until it becomes completely dense in the last layer. This is caused by enlarging the receptive field through convolutions and pooling operations. On the other hand, all the feature maps in the decoder part are dense.

The percentage of non-zero values at each encoder level can be estimated by calculating the receptive field size. The SC-UNET concatenates feature maps at the level where the sparseness is lower than a certain value.

The network has five downsampling and upsampling layers (levels) respectively in the encoder and decoder. A similarity metric is used to calculate the similarity between analogous layers, using the ratio of the inner product over the L2 norm of the feature maps. The paper shows that the similarity increases as the spatial resolution of the feature maps decreases.

![Untitled](Color%20Image%20Generation%20from%20LiDAR%20Reflection%20Data%20%2021ae8ab4fe66484f809c9e6bd127bec1/Untitled.png)
## Results
**Observation 1.** At a low level, a small amount of valid information can be transferred to the decoder side via concatenation. For example, the amount of feature map data to be transferred is very limited if only level 1 is concatenated.

**Observation 2.** At a high level, the encoder feature map has high sparseness. For example, the structure having a single connection at level 4 is expected to have limited performance due to the small number of valid pixels.

**Observation 3.** At a low level, the similarity between feature maps of the encoder and decoder parts increases. For example, the structure with a single connection at level 4 is expected to have limited performance due to the very different characteristics between the encoder and decoder feature maps.

Emphasises the importance of taking into account sparseness when selecting connections in Unet architectures.
## Dataset
KITTI dataset

recorded images under heavy shadows were not included to enable shadow-free colour image generation

signal-to-noise ratio (PSNR) and structural similarity (SSIM) were used to evaluate the image quality between the generated and target colour images