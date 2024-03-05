## Approach
StarGAN paper uses multi-domain image to image translation using only one Generator and Discriminator for any class.

In addition to the input channel, the paper also uses a channel of the class label spatially replicated to the image size.

Apart from identifying if the generated image is fake or not, the discriminator uses a classifier output to check if the generated class is the right one (the generator learns to create an image that is close to the target label).

Also uses cycle-consistency loss as CycleGAN