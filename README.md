# Speculator
[![Static Badge](https://img.shields.io/badge/arXiv-1911.11778-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/1911.11778)
[![DOI](https://img.shields.io/badge/DOI-10.3847%2F1538--4365%2Fab917f-%23fab70c?logo=doi&logoColor=%23fab70c)](https://doi.org/10.3847/1538-4365/ab917f)

This repository contains the code for neural network emulation of stellar population synsthesis (SPS) models for galaxy spectra, originally published in Alsing et. al ([2020](https://ui.adsabs.harvard.edu/abs/2020ApJS..249....5A/abstract)). If you use this code, kindly cite that paper. The parent fork at [justinalsing/speculator](https://github.com/justinalsing/speculator) is the home for the original `tensorflow` version of the codebase.

## Installation

You can install the code with pip: `pip install git+https://github.com/justinalsing/speculator.git`

The code is in python3 and has the following dependencies:<br>
[tensorflow](https://www.tensorflow.org) (>2.0) <br> 
[scikit-learn](https://scikit-learn.org/stable/)<br> 
[numpy](https://numpy.org)<br> 

## Demo

A basic demo of loading and calling a pre-trained model (Prospector-alpha) can be found in `examples/speculator_demo.ipynb`. For training your own model, you can use the template given in `examples/speculator_training_demo.ipynb`.

## Updates and Collaboration

This `tensorflow` version of the code is no longer being actively maintained, but will remain here for those still using it. The `torch` version developed in the `torch` branch of [justinalsing/speculator](https://github.com/justinalsing/speculator) will continue to be developed as the default branch of [Cosmo-Pop/speculator](https://github.com/Cosmo-Pop/speculator).
