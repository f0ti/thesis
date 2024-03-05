## Motivation
The weak point of StarGAN that this paper is trying to overcome is the diversity. StarGAN learns a deterministic mapping per each domain, which does not capture the multi-model nature of the data distribution. This comes from the fact that each domain is indicated by a predetermined label which the generator receives as input.
The paper proposes a style network encoder in order to diversify the generated output for a domain.
## Approach
