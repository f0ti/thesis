## Motivation
The creation of training data is expensive, so there have been proposed methods for producing synthetic training data. Models trained naively on synthetic data do not typically generalize to real images. To overcome this, the paper proposes transfering the knowledge learned from a source domain, for which the label data exists, to a target domain for there are no labels available.

**Advantages**:
- the architecture is task-independent, so no re-train is needed for domain adaption.
- generalization across label spaces, handles cases where the target label space at test time differs from the label space at training time
- data augmentation, creates a virtually unlimited stochastic samples
## Comments
A similar work to this paper, is [Coupled Generative Adversarial Networks](https://arxiv.org/pdf/1606.07536.pdf) which uses a pair of coupled GANs, one for the source and one for the target domain, whose generators share their high-layer weights and whose discriminators share their low-layer weights.
Style transfer is an interesting method to be explored as well, in which the style of one image is transferred to another while holding the content fixed.
## Approach
As stated in the CycleGAN paper, this model forces the output to be close to the input based on a class label space.
Given a labeled dataset in a source domain and an unlabeled dataset in a target domain, our goal is to train a classifier on data from the source domain that generalizes to the target domain.
Differently from the standard GAN formulation, in which the generator is conditioned only on a noise vector, the proposed model is conditioned on both noise vector and image from the source domain.
In addition to the discriminator, a classifier $T$ is introduced which assigns task-specific labels $\hat{y}$ to images $x$ coming from the generator and the target domain.
## Loss functions
- Adversarial loss
- Classifier $T$ loss
- Content-similarity loss, penalizes large differences between source and generated images for foreground pixels only

![[Pasted image 20240227165339.png]]
