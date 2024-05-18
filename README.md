# GEOL0069_final

## Introduction
This is the final assignment for GEOL0069 Artificial Intelligence for Earth Observation (AI4EO). It uses Sentinel-3 OLCI (Ocean and Land Colour Instrument) data over Parque Nacional Amboró in Bolivia in the South American continent for cloud classification. Masks are created for each of the 3 region of interest (roi) using IRIS (Intelligently Reinforced Image Segmentation), where image pixels are being classified as cloud and non-cloud, and they represent the "ground truth".
Machine learning models (Convolutional Neural Networks (CNN), Random Forest (RF) and Vision Transformer (ViT)) are then created, using one of the masks. The models are then applied to the 3 roi, and confusion matrices are calculated and created.

## Copernicus Browser
The Sentinel-3 data is downloaded from Copernicus Browser (https://browser.dataspace.copernicus.eu/). 
![image](https://github.com/eunicewly/GEOL0069_final/assets/159627060/2fdbe29d-e72b-4c92-864a-1f3663a8896e)

Manually, the 

<img width="332" alt="image" src="https://github.com/eunicewly/GEOL0069_final/assets/159627060/413aa4bc-a5a3-4732-b3a8-34b510c527d4">

<img width="328" alt="image" src="https://github.com/eunicewly/GEOL0069_final/assets/159627060/541e7b53-cabf-4f39-a7f3-074e71dc27cf">

<img width="954" alt="image" src="https://github.com/eunicewly/GEOL0069_final/assets/159627060/8a03e1b2-fcc3-4404-b97a-ff587085be12">



<img width="952" alt="image" src="https://github.com/eunicewly/GEOL0069_final/assets/159627060/42e4dd93-804a-46d5-838e-9d1f1695dcc7">


## Docker

## IRIS
