# σ-CFDM (Closed-Form Diffusion Models)

Code implementation of **σ-CFDMs**.
>Score-based generative models (SGMs) sample from a target distribution by iteratively transforming noise using the score function of the perturbed target.
>For any finite training set, this score function can be evaluated in closed form, but the resulting SGM memorizes its training data and does not generate novel samples.
>In practice, one approximates the score by training a neural network via score-matching.
>The error in this approximation promotes generalization, but neural SGMs are costly to train and sample, and the effective regularization this error provides is not well-understood theoretically.
>In this work, we instead explicitly smooth the closed-form score to obtain an SGM that generates novel samples without training.
>We analyze our model and propose an efficient nearest-neighbor-based estimator of its score function.
>Using this estimator, our method achieves sampling times competitive with neural SGMs while running on consumer-grade CPUs.
>[Scarvelis et al. (2023)](https://arxiv.org/abs/2310.12395)
