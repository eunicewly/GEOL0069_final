# GEOL0069_final

## Introduction
This is the final assignment for GEOL0069 Artificial Intelligence for Earth Observation (AI4EO). It uses Sentinel-3 OLCI (Ocean and Land Colour Instrument) data over Parque Nacional Amboró in Bolivia in the South American continent for cloud classification. Masks are created for each of the 3 region of interest (roi) using IRIS (Intelligently Reinforced Image Segmentation), where image pixels are being classified as cloud and non-cloud, and they represent the "ground truth".
Machine learning models (Convolutional Neural Networks (CNN), Random Forest (RF) and Vision Transformer (ViT)) are then created, using one of the masks. The models are then applied to the 3 roi, and confusion matrices are calculated and created.

## Copernicus Browser
The Sentinel-3 data is downloaded from Copernicus Browser (https://browser.dataspace.copernicus.eu/). 
![image](https://github.com/eunicewly/GEOL0069_final/assets/159627060/2fdbe29d-e72b-4c92-864a-1f3663a8896e)


## Docker

## IRIS
