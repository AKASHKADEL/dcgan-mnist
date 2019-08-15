# DCGAN 

PyTorch implementation of DCGAN on MNIST dataset. The implementation primarily follows the paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf).

# Introduction

Deep Convolutional GAN is one of the most coolest and popular deep learning technique. It is a great improvement upon the [original GAN network](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) that was first introduced by Ian Goodfellow at NIPS 2014. DCGAN uses the same framework of generator and discriminator. This is analogous to solving a two player minimax game: Ideally the goal of the discriminator is to be very sharp in distinguishing between the real and fake data, whereas, generator aims at faking data in such a way that it becomes nearly impossible for the discriminator to classify it as a fake. The below gif shows how quickly dcgan learns the distribution of mnist and generates real looking digits.

[](https://github.com/AKASHKADEL/dcgan-mnist/blob/master/results/fixed_noise/animated.gif)

