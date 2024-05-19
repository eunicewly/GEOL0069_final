# GEOL0069_final

## Introduction
This is the final assignment for GEOL0069 Artificial Intelligence for Earth Observation (AI4EO). It uses Sentinel-3 OLCI (Ocean and Land Colour Instrument) data over the Parque Nacional Amboró in Bolivia in the South American continent for cloud classification. Image pixels in 3 region of interest (roi) are being classified as cloud and non-cloud using IRIS (Intelligently Reinforced Image Segmentation), and they represent the "ground truth".
Machine learning models (Convolutional Neural Networks (CNN), Random Forest (RF) and Vision Transformer (ViT)) are then created. The models are then applied to the 3 roi, and confusion matrices are calculated and created.

## Copernicus Browser
The Sentinel-3 data is downloaded from Copernicus Browser (https://browser.dataspace.copernicus.eu/). Manually, unders search, Sentinel-3 OLCI Level 1 EFR data from the 10th May 2024 is selected.

<img width="332" alt="image" src="https://github.com/eunicewly/GEOL0069_final/assets/159627060/413aa4bc-a5a3-4732-b3a8-34b510c527d4">

<img width="328" alt="image" src="https://github.com/eunicewly/GEOL0069_final/assets/159627060/541e7b53-cabf-4f39-a7f3-074e71dc27cf">

\
This reveals that the file S3B_OL_1_EFR____20240510T140254_20240510T140554_20240510T154337_0180_093_010_3240_PS2_O_NR_004.SEN3 is available (shown below; the first search result):
\
<img width="800" alt="image" src="https://github.com/eunicewly/GEOL0069_final/assets/159627060/42e4dd93-804a-46d5-838e-9d1f1695dcc7">
\
\
The true colour image of this data:
![image](https://github.com/eunicewly/GEOL0069_final/assets/159627060/2fdbe29d-e72b-4c92-864a-1f3663a8896e)
\
This file is downloaded from Copernicus browser directly and uploaded to Google Drive.


## Docker
Docker is being utlised to run IRIS in a containerised environment, to simplify the installation and setting up of IRIS. Docker is installed following the guide from https://docs.docker.com/get-docker/. 

1. In the terminal or command prompt on your computer, with your docker opened, type
```
docker pull totony4real/iris:1.0
```
which will allow you to run IRIS in a Docker container and access the Iris web interface.

2. Then type
```
docker run -p 80:5000 -v "path_to_file":/dataset/ --rm -it totony4real/iris:1.0 label /dataset/config.json
```
with the "path_to_file" replaced by your actual path in which the Sentinel-3 data is saved.

3. After that, open http://localhost:80 on your browser, 

## IRIS
