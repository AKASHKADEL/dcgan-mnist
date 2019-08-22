# DCGAN 

A small PyTorch tutorial for DCGAN on MNIST dataset. The implementation primarily follows the paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf).

# Introduction

Deep Convolutional GAN is one of the most coolest and popular deep learning technique. It is a great improvement upon the [original GAN network](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) that was first introduced by Ian Goodfellow at NIPS 2014. (DCGANs are much more stable than Vanilla GANs) DCGAN uses the same framework of generator and discriminator. This is analogous to solving a two player minimax game: Ideally the goal of the discriminator is to be very sharp in distinguishing between the real and fake data, whereas, generator aims at faking data in such a way that it becomes nearly impossible for the discriminator to classify it as a fake. The below gif shows how quickly dcgan learns the distribution of mnist and generates real looking digits.

![](https://github.com/AKASHKADEL/dcgan-mnist/blob/master/results/fixed_noise/animated.gif)

# Quick Start

To get started and to replicate the above result, follow the instructions in this section. This wil allow you to train the model from scratch and help produce basic visualizations. 

## Dependencies:

* Python 3+ distribution
* PyTorch >= 1.0

Optional:

* Matplolib and Imageio to produce basic visualizations.
* Cuda >= 10.0

Once everything is installed, you can go ahead and run the below command to train a model on 100 Epochs and store the sample outputs from generator in the ```results``` folder.

```python main.py --num-epochs 100 --output-path ./results/ ```

You can also generate sample output using a fixed noise vector (It's easier to interpret the output on a fixed noise. Ex: the above gif), use this

```python main.py --num-epochs 100 --output-path ./results/ --use-fixed ```

You can change the model setting by playing with the learning rate, num_epochs, batch size, etc

## Outputs

The above code will store 100 images in the folder ```./results/fixed_noise```, each storing the output after every epoch. Also, the imageio library will then take these 100 images a create a gif out of it with fps=5. The final gif will be stored in the same folder. ie., ```./results/fixed_noise/animated.gif```

# References:

[1] https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf <br>
[2] https://arxiv.org/pdf/1511.06434.pdf <br>
[3] https://github.com/soumith/ganhacks <br>
[4] https://medium.com/activating-robotic-minds/up-sampling-with-transposed-convolution-9ae4f2df52d0 <br>
