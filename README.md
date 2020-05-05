# speculator
Neural network emulation of Stellar Population Synthesis (SPS) models for galaxy spectra and photometry.

This repository contains the neural SPS emulation code associated with the paper [Alsing et. al 2019, arXiv 1911.11778](https://arxiv.org/abs/1911.11778)

# dependencies

The code is in python3 and has the following dependencies:<br>
[tensorflow](https://www.tensorflow.org) (>2.0) <br> 
[scikit-learn](https://scikit-learn.org/stable/)<br> 
[numpy](https://numpy.org)<br> 

# demo

A basic demo of loading and calling a pre-trained model (Prospector-alpha) can be found in `speculator_demo.ipynb`

# updates and collaboration

We will be expanding this repository with additional models and demonstrations of how to train your own models. Meanwhile, if you require a particular model or would like to collaborate on building an SPS emulator for a particular case/project, feel free to get in touch at justin.alsing@fysik.su.se
