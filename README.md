# Unsupervised-ML-on-Strong-Gravitational-Lensing-Detection-using-Convolutional-Autoencoder
This toy code is simplified for demonstration based on the codes used in the paper: [Identifying Strong Lenses with Unsupervised Machine Learning using Convolutional Autoencoder, Cheng et al.](https://arxiv.org/abs/1911.04320)".  

The goal of this is to apply an unsupervised machine leanring techniques consisting of **Convolutional Autoencoder** and **Bayesian Gaussian Mixture Model** to identify galaxy-galaxy strong lensing systems.
- Feature Extraction using **Convolutional Autoencoder** (CAE.py)
- Clustering data at the high-dimensional feature space by **Bayesian Gaussian Mixture model** (BGMM.py)

100 images attached in this repo are in linear scale and after the denoise process by the CAE with a simplified architecture (no dense layer, details are shown in paper attached above). The complete original simulated data used and the classification table can be found in [Strong Lenses Finding Challenge v1](http://metcalf1.difa.unibo.it/blf-portal/gg_challenge.html). 
