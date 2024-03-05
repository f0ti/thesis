## Motivation
Given two related domains $S$ and $T$, the goal is to learn a generative function $f$, that maps an input sample from $S$ to the domain $T$, such that the function $f$ stays the same, independent from either of the inputs in both domains.

$G: S \rightarrow T$ such that $f(x) \sim f(G(x))$
## Approach
The paper proposes Domain Transfer Network, with a multi-class GAN loss, f-constancy component, and a regularizing component that encourages $G$ to map samples from $T$ to themselves.
The Networks seems to be a convential GAN with only modifications on the loss functions.
![[Pasted image 20240229113109.png]]
## Loss functions
- Adversarial loss of GAN ($L_{GAND}$ and $L_{GANG}$)
- $L_{CONST}$ keep $f(G(s))$ and $f(s)$ identical, so they have the same encoded features
- $L_{TID}$ wants $G$ to remain the identity mapping when it takes in some $t$ from the target domain, as seen from the diagram measures between the same input and target from the same target domain
- $L_{TV}$ to smooth the resulting image
## Datasets
- The paper performs the experiment on transfering SVHN to MNIST, and trasferring unlabeled face images to a space of emoji images