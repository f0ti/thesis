## Motivation
Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. For many tasks the paired training data will not be available.
The goal is to translate an image from a source domain X to a target domain Y in the absence of these paired examples, such that the distribution of images from $G(x)$ is indistinguishable from the distribution Y using an adversarial loss.
This paper introduces an inverse mapping $F: Y \rightarrow x$  and a cycle consistency loss.
## Approach
Capturing special characteristics of one image collection and figuring out how these characteristics could be translated into the other image collection, all in the absence of any paired training examples.

We also make the assumption that there exists some underlying relationship between the domains.

The optimal $G$ translates the domain $X$ to a domain $\hat{Y}$ that have the same distribution. However, such a translation does not guarantee that an individual input $x$ and output $y$ are paired in a meaningful way, because there are infinitely many mappings $G$ that will induce the same distribution over $y$.

Cycle consistency - if $G: X \rightarrow Y$ and $F: Y \rightarrow X$, then $G$ and $F$ should be inverses of each other, and both mappings should be bijections.

The focus of the paper is to learn the mapping between two image collections (domains), not just two images.

![[architecture.png]]
The discriminator $D_x$ goal is to distinguish between images $\{x\}$ and translated images $F\{y\}$, and the goal of discriminator $D_y$ is to distinguish between $\{y\}$ and translated images $G\{x\}$.
## Loss function
- **Adversarial loss** - matching the distribution of generated images to the data distribution in the target domain, the discriminator aims to distinguish between translated samples $G(x)$ and real samples $y$. The generator is trying to produce outputs that are identically distributed as $X$ and $Y$. 
- **Cycle consistency loss** - prevent the learned mappings $G$ and $F$ from contradicting each other, it introduced a constraint on the space of mapping functions
	- *forward cycle consistency* - the image translation cycle $G$ should bring $x$ back to the original image
	  $x \rightarrow G(x) \rightarrow F(G(x)) \approx x$ 
	- *backward cycle consistency* - the backward image translation cycle $F$ should bring $y$ to the generated image
	  $y \rightarrow F(y) \rightarrow G(F(y)) \approx y$

The negative log likelihood on the adversarial loss is replaced with least-squares loss.
The reduce model oscillation, the discriminator is updated using a history of generated images buffer rather than the ones produced by the latest generators.
## Results
Qualitative evaluation using Amazon Mechanical Turk, could foold at least a quarter of the participants.
## Dataset
Cityscape Dataset labels to photo
Google Maps maps to aerial photo
## Comments
The paper mentions that their formulation is not task dependent, differently from the previous approaches which encourage the input and the output to share specific "content" features. In our case, we need to maintain the geometric information (shapes and edges). The papers below force the the output to be close to the input in a predefined metric space, such as [class label space](https://arxiv.org/pdf/1612.05424.pdf), [image pixel space](https://arxiv.org/pdf/1612.07828.pdf), and [image feature space](https://arxiv.org/pdf/1611.02200.pdf).

This model can be seen as training two autoencoders. They map an image to itself via an intermediate representation that is a translation of the image into another domain.

When the real samples of the target domain are provided as the input to the generator, then we could use the identity mapping, which helps on preserving the color composition between input and output.