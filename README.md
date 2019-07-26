# SeqGAN

## Requirements: 
* **Tensorflow r1.0.1**
* Python 2.7
* CUDA 7.5+ (For GPU)

## Introduction
Apply Generative Adversarial Nets to generating sequences of discrete tokens.

The code is based on [the code of SeqGAN](https://github.com/LantaoYu/SeqGAN) but replace the generation data part with custom corpus.

The research paper is [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](http://arxiv.org/abs/1609.05473) .

To run the experiment with default parameters:

```
$ python sequence_gan.py
```
You can change the all the parameters in `sequence_gan.py`.

The experiment has two stages:

- In the first stage, use the positive data provided by the oracle model and Maximum Likelihood Estimation to perform supervise learning. 
- In the second stage, use adversarial training to improve the generator.