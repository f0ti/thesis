## Motivation
While synthesizing images, there is still a gap between synthetic and real image distributions. If a model is trained on these data, then it learns the features that are only present in these synthetic data.
To reduce this gap, a Simulated+Unsupervised learning is proposed, where the goal is to learn a model to improve the realism of a simulator's output using unlabeled real data, while preserving the annotation information from the simulator. The GAN model used is not based on noise vector, but instead on synthetic images as inputs.
## Approach
SimGAN refines synthetic images from a simulator using the refiner network.
## Loss functions
**Adversarial loss with self-regularization**
It is used to add realism to the synthetic images, by training a refiner network $R$ which tries to fool discriminator $D$, that is trained to classify images as real or refined. It is also important for the generated images from $R$ to preserve the annotation information  of the simulator. For this purpose, the paper proposes using a self-regularization loss that minimizes per-pixel difference between a feature transform of the synthetic and refined images. This feature transform can be an identity map, image derivatives, mean of color channels.

**Local Adversarial Loss**
The refiner network should learn to model the real image characteristics without introducing any artifacts. Usually when a single strong discriminator network is trained, the refiner network tends to over-emphasize certain image features and that leads to drifting and produces artifacts.
The assumption is that any local patch sampled from the refined image should have similar statistics to a real image patch. So, rather than defining a global discriminator network, a discriminator network is proposed that classifies all local image patches separately.

![[Pasted image 20240227180936.png]]

**Updating Discriminator using a History of Refined Images**
Instead of updating the discriminator only on the current mini-batch, a history of refined images is used. The assumption made is that any image generated from the refiner network at any time should be assigned as "fake" from the discriminator. An image buffer with size $B$ is kept and $b/2$ images are generated from the refiner network and then randomly replaces in the buffer. 

![[Pasted image 20240227173000.png]]
## Results
Really good results on Gaze Dataset and Hand Pose Estimation