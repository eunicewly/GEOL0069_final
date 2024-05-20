# GEOL0069_final

## Introduction
This is the final assignment for GEOL0069 Artificial Intelligence for Earth Observation (AI4EO). It uses Sentinel-3 OLCI (Ocean and Land Colour Instrument) data over the Parque Nacional Amboró in Bolivia in the South American continent for cloud classification. Image pixels in 3 region of interest (roi) are being classified as cloud and non-cloud using IRIS (Intelligently Reinforced Image Segmentation), and they represent the "ground truth".
Machine learning models (Convolutional Neural Networks (CNN), Random Forest (RF) and Vision Transformer (ViT)) are then created. The models are then applied to the 3 roi, and confusion matrices are calculated and created.

This README.md describes the methods of (i) downloading data from Copernicus browser, (ii) data pre-processing for the creation of images to be loaded into IRIS, and (iii) utilising docker and IRIS to classify image pixels. The details on the codes to create the models and classify images is available in the GEOL0069_final_cloud_classification.ipynb in this repository. 

## Copernicus Browser
The Sentinel-3 data is downloaded from Copernicus Browser (https://browser.dataspace.copernicus.eu/). Manually, under search, Sentinel-3 OLCI Level 1 EFR data from the 10th May 2024 is selected.

<img width="332" alt="image" src="https://github.com/eunicewly/GEOL0069_final/assets/159627060/413aa4bc-a5a3-4732-b3a8-34b510c527d4">

<img width="328" alt="image" src="https://github.com/eunicewly/GEOL0069_final/assets/159627060/541e7b53-cabf-4f39-a7f3-074e71dc27cf">

\
This reveals that the file S3B_OL_1_EFR____20240510T140254_20240510T140554_20240510T154337_0180_093_010_3240_PS2_O_NR_004.SEN3 is available (shown below; the first search result), and it is over the Parque Nacional Amboró in Bolivia in the South American continent:
\
<img width="800" alt="image" src="https://github.com/eunicewly/GEOL0069_final/assets/159627060/42e4dd93-804a-46d5-838e-9d1f1695dcc7">
\
\
The true colour image of this data:
![image](https://github.com/eunicewly/GEOL0069_final/assets/159627060/2fdbe29d-e72b-4c92-864a-1f3663a8896e)
\
This file is downloaded from Copernicus browser directly and uploaded to Google Drive.


## Data Pre-processing
Opening a Jupyter notebook (or follow the GEOL0069_final_cloud_classification.ipynb in this repository), run the code to:

1. Convert the Sentinel-3 data from netCDF4 format to numpy arrays (.npy), and split them into 5 chunks such that it is possible for IRIS to display them in a single interface view (convert all "path_to_be_added" to the actual path where your data is located at):
```
! pip install rasterio
! pip install netCDF4
```
```
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
import netCDF4
import numpy as np
import re
from sklearn.feature_extraction import image

# Define the path to the main folder where your data is stored.
# You need to replace 'path/to/data' with the actual path to your data folder.
main_folder_path = 'path_to_be_added'
# main_folder_path = './'
# This part of the code is responsible for finding all directories in the main_folder that end with '.SEN3'.
# '.SEN3' is the format of the folder containing specific satellite data files (in this case, OLCI data files).
directories = [d for d in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, d)) and d.endswith('004.SEN3')] #load OLCI imagery

# Loop over each directory (i.e., each set of data) found above.
for directory in directories:
    # Construct the path to the OLCI data file within the directory.
    # This path is used to access the data files.
    OLCI_file_p = os.path.join(main_folder_path, directory)

    # Print the path to the current data file being processed.
    # This is helpful for tracking which file is being processed at any time.
    print(f"Processing: {OLCI_file_p}")

    # Load the instrument data from a file named 'instrument_data.nc' inside the directory.
    # This file contains various data about the instrument that captured the satellite data.
    instrument_data = netCDF4.Dataset(OLCI_file_p + '/instrument_data.nc')
    solar_flux = instrument_data.variables['solar_flux'][:]  # Extract the solar flux data.
    detector_index = instrument_data.variables['detector_index'][:]  # Extract the detector index.

    # Load tie geometries from a file named 'tie_geometries.nc'.
    # Tie geometries contain information about viewing angles, which are important for data analysis.
    tie_geometries = netCDF4.Dataset(OLCI_file_p + '/tie_geometries.nc')
    SZA = tie_geometries.variables['SZA'][:]  # Extract the Solar Zenith Angle (SZA).

    # Create a directory for saving the processed data using the original directory name.
    # This directory will be used to store output files.
    save_directory = os.path.join('/content/drive/MyDrive/GEOL0069/final/output', directory)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # This loop processes each radiance band in the OLCI data.
    # OLCI instruments capture multiple bands, each representing different wavelengths.
    OLCI_data = []
    for Radiance in range(1, 22):  # There are 21 bands in OLCI data.

        Rstr = "%02d" % Radiance  # Formatting the band number.
        solar_flux_band = solar_flux[Radiance - 1]  # Get the solar flux for the current band.

        # Print information about the current band being processed.
        # This includes the band number and its corresponding solar flux.
        print(f"Processing Band: {Rstr}")
        print(f"Solar Flux for Band {Rstr}: {solar_flux_band}")

        # Load radiance values from the OLCI data file for the current band.
        OLCI_nc = netCDF4.Dataset(OLCI_file_p + '/Oa' + Rstr + '_radiance.nc')
        radiance_values = np.asarray(OLCI_nc['Oa' + Rstr + '_radiance'])

        # Initialize an array to store angle data, which will be calculated based on SZA.
        angle = np.zeros_like(radiance_values)
        for x in range(angle.shape[1]):
            angle[:, x] = SZA[:, int(x/64)]

        # Calculate the Top of Atmosphere Bidirectional Reflectance Factor (TOA BRF) for the current band.
        TOA_BRF = (np.pi * radiance_values) / (solar_flux_band[detector_index] * np.cos(np.radians(angle)))

        # Add the calculated TOA BRF data to the OLCI_data list.
        OLCI_data.append(TOA_BRF)

    reshaped_array = np.moveaxis(np.array(OLCI_data), 0, -1)
    OLCI_coord = netCDF4.Dataset(OLCI_file_p + '/geo_coordinates.nc')
    OLCI_lon=OLCI_coord['longitude']
    OLCI_lat=OLCI_coord['latitude']

    # Reshape the OLCI_data array for further analysis or visualization.
    reshaped_array = np.moveaxis(np.array(OLCI_data), 0, -1)
    print("Reshaped array shape:", reshaped_array.shape)

        # Split the reshaped array into smaller chunks along the second dimension.
        # This can be useful for handling large datasets more efficiently.
    split_arrays = np.array_split(reshaped_array, 5, axis=1)

        # Save each chunk of data separately.
        # This is helpful for processing or analyzing smaller portions of data at a time.
    for i, arr in enumerate(split_arrays):
           print(f"Chunk {i+1} shape:", arr.shape)
           save_path = os.path.join(save_directory, f"chunk_{i+1}_band_{Rstr}.npy")
           np.save(save_path, arr)
           print(f"Saved Chunk {i+1} for Band {Rstr} to {save_path}")

```
Five chunk images are then created (chunk_1_band_21.npy, chunk_2_band_21.npy and so on). Chunk 3 is shown below:
\
![image](https://github.com/eunicewly/GEOL0069_final/assets/159627060/9bb47625-9cb1-403a-b303-a294daa3ec86)

\
\
2. Extract three region of interest (roi) from the 3rd chunk image (saved as image_chunk3):
```
x1, y1, x2, y2 = [100,700,300,1000] # roi a
x1b, y1b, x2b, y2b = [100, 400, 300, 700] #roi b
x1c, y1c, x2c, y2c = [100, 1000, 300, 1300] #roi c

roi3a = image_chunk3[y1:y2, x1:x2]
roi3b = image_chunk3[y1b:y2b, x1b:x2b]
roi3c = image_chunk3[y1c:y2c, x1c:x2c]
```
\
\
3. Visuallising the three roi:
```
# Extract channels 1, 2, and 3
channel_3a = roi3a[:,:,6]  # 0-based indexing for the first channel
rgb_image3a = np.stack([channel_3a], axis=-1) # You can add more channels if you want
channel_3b = roi3b[:,:,6]  # 0-based indexing for the first channel
rgb_image3b = np.stack([channel_3b], axis=-1)
channel_3c = roi3c[:,:,6]  # 0-based indexing for the first channel
rgb_image3c = np.stack([channel_3c], axis=-1)

# Plotting the masks area out
plt.figure(figsize=(11, 11))
plt.subplot(1, 3, 1)
plt.imshow(rgb_image3a)
plt.subplot(1, 3, 2)
plt.imshow(rgb_image3b)
plt.subplot(1, 3, 3)
plt.imshow(rgb_image3c)
plt.axis('off')
plt.show()
```
![image](https://github.com/eunicewly/GEOL0069_final/assets/159627060/8981b646-ef0b-4826-8270-a6c84cfe6aaa)
\
\
4. Save the roi onto your computer, preparing for loading into IRIS for cloud classification:
```
path = "path_to_be_added"
np.save(path + 'roi3a.npy', roi3a)
np.save(path + 'roi3b.npy', roi3b)
np.save(path + 'roi3c.npy', roi3c)
```



## Docker
Docker is being utlised to run IRIS in a containerised environment, to simplify the installation and setting up of IRIS. Thus, before running IRIS, docker is installed. Docker is installed following the guide from https://docs.docker.com/get-docker/. 

1. Have the config.json file (available in this respository) saved in the folder you saved the three roi images.
   

2. In the terminal or command prompt on your computer, with your docker engine running, type
```
docker pull totony4real/iris:1.0
```
(keep the spacing) which will allow you to run IRIS in a Docker container and access the Iris web interface.

3. Then type
```
docker run -p 80:5000 -v "path_to_file":/dataset/ --rm -it totony4real/iris:1.0 label /dataset/config.json
```
with the "path_to_file" replaced by your actual path in which the Sentinel-3 data is saved.

For example:

<img width="758" alt="Screenshot 2024-05-19 154425" src="https://github.com/eunicewly/GEOL0069_final/assets/159627060/31103d6c-3c2b-441e-97ae-f395fd8e8d85">

\
4. After that, open http://localhost:80 on your browser, create your own account by typing your username and password.

<img width="418" alt="image" src="https://github.com/eunicewly/GEOL0069_final/assets/159627060/485b9df3-4207-44fc-8d05-ff249fbd7e07">


## IRIS - Intelligently Reinforced Image Segmentation
IRIS is a tool for supervised image segmentation of satellite imagery, in which the user manually classify some image pixels and AI (gradient boosted decision tree) the rest. It was designed to accelerate the creation of machine learning training datasets. 

Image pixels are classified into "cloud" and "non-cloud" class:

<img width="587" alt="image" src="https://github.com/eunicewly/GEOL0069_final/assets/159627060/eda5ef6f-dd82-49cd-9772-42d52f11b204">

\
One of the classification results:


<img width="954" alt="Screenshot 2024-05-14 235405" src="https://github.com/eunicewly/GEOL0069_final/assets/159627060/bd857f75-3395-4845-8707-04c55967daa5">


\
All three roi images (roi3a, roi3b, roi3c) are classified. These classification represents the ground-truth, and are saved into the computer (as masks) and loaded into the Jupyter notebook:
![image](https://github.com/eunicewly/GEOL0069_final/assets/159627060/d1ae9330-16ed-4b9d-a32f-4c9702b13957)



## Model creation
(The details and codes can be found in GEOL0069_final_cloud_classification.ipynb in this repository)

Training and testing data are then created from roi3a, for model (Convolutional Neural Networks (CNN), Random Forest (RF) and Vision Transformer (ViT)) creation. These three models are then applied to the three roi. The classification and confusion matrices are as follows:


### For CNN:
\
![image](https://github.com/eunicewly/GEOL0069_final/assets/159627060/90dda281-a335-4739-b802-100939ccf578)
\
\
<img width="544" alt="image" src="https://github.com/eunicewly/GEOL0069_final/assets/159627060/e2cd3463-5ac8-4a89-aaef-5ecf5f77ab9a">


### For RF:
![image](https://github.com/eunicewly/GEOL0069_final/assets/159627060/baa38682-5b66-4358-aeb7-abdcde147c55)
\
\
<img width="534" alt="image" src="https://github.com/eunicewly/GEOL0069_final/assets/159627060/7656f197-7fdb-4b19-9224-2aa5e2f5e1b1">


### For ViT:
![image](https://github.com/eunicewly/GEOL0069_final/assets/159627060/309bf68c-f8ab-46b3-a7e9-3f128ac5576e)
\
\
<img width="537" alt="image" src="https://github.com/eunicewly/GEOL0069_final/assets/159627060/79df5d64-6a1f-4899-813e-196cc10202e4">
\
\
\
Full region rollout has been attempted but without paying Google (buying Google colab pro, for the extra RAM), it is not possible...


