# Custom ProGAN Implementation

This repository contains my custom implementation of the Progressive GAN (ProGAN) with tailored modifications in hyperparameters (because I don't have superfast GPUs). ProGAN is renowned for its progressive training strategy that enables the generation of high-resolution images with remarkable quality.

## Table of Contents

- [Overview](#overview)
- [Modifications](#modifications)
- [Results](#results)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Overview

The authors describe a new training methodology for generative adversarial networks. The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, they add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality, e.g., CelebA images at 1024^2. They also propose a simple way to increase the variation in generated images, and achieve a record inception score of 8.80 in unsupervised CIFAR10. Additionally, they describe several implementation details that are important for discouraging unhealthy competition between the generator and discriminator. Finally, they suggest a new metric for evaluating GAN results, both in terms of image quality and variation. As an additional contribution, they construct a higher-quality version of the CelebA dataset.

## Details

![Model Training Scheme](images/training.png)

## Results

Here are some of the results obtained
![Results](results/64_examples.png)

### Performance Metrics

If applicable, include metrics and evaluation results:

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- FID (Frechet Inception Distance)
- Other relevant metrics

## Getting Started

Provide detailed instructions on how to set up and run your customized ProGAN. Include installation steps, prerequisites, and any additional setup requirements.

```bash
# Installation instructions
