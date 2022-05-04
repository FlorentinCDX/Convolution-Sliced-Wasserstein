# Revisiting Sliced Wasserstein on Images: From Vectorization to Convolution

This repository provides an unofficial implementation of the paper [Revisiting Sliced Wasserstein on Images: From Vectorization to Convolution](https://nhatptnk8912.github.io/Revisiting_Sliced_Wasserstein_Arxiv.pdf) 

I reimplemented the three types of convolutional slicer:
- Convolutional Base Slicer
- Convolutional Stride Slicer
- Convolutional Dilatation Slicer

As well as a Convolution Sliced Wasserstein class for every slicer. I also provide a summary of the paper in conv-SW.pdf. My implementation is not official and only work for even dimentional inpu data.

## Installation

This projet will required the folowwing depedencies :
- torch
- torchvision (to run the test on images)

## Content and reproduction

The python script random_convolution.py contains all the classes and method. The notebook test_CSW.ipynb contains the reproducable on toy data and images from celebA dataset.

