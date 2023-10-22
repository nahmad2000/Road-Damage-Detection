# Real-Time Cracks Detection and Severity Classification System

## Background
Road infrastructure is prone to deterioration, and the presence of cracks poses significant safety risks. Early detection of these road defects is essential to enhance safety, reduce maintenance costs, and eliminate the inefficiencies associated with manual inspections. Innovative solutions, such as Industrial Internet of Things (IIoT) technologies, offer promising prospects for addressing these challenges.


## Dataset
- The dataset used is the .
- The dataset can be downloaded from .
- This dataset is a mix of 7 datasets gathered from different cities: 
  - China_Drone
  - China_MotorBike
  - Czech
  - India
  - Japan
  - Norway
  - United_States


Our work utilizes the Road Damage Detection 2022 (RDD2022) dataset. This comprehensive dataset can be accessed via this [link](https://doi.org/10.48550/arXiv.2209.08538), and the data can be downloaded from this [source](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547?file=38030820).

RDD2022 combines data from seven distinct datasets collected across various cities, including:
- China_Drone
- China_MotorBike
- Czech
- India
- Japan
- Norway
- United_States

Here is an overview of the dataset structure:


```plaintext
\---RDD2022
    +---China_Drone
    |   \---train
    |       +---annotations
    |       |   \---xmls
    |       \---images
    +---China_MotorBike
    |   +---test
    |   |   \---images
    |   \---train
    |       +---annotations
    |       |   \---xmls
    |       \---images
    ...



## Repository Contents:

1. preprocessing.py
This script contains a set of functions for preprocessing the RDD2022 dataset, making it ready for machine learning.

### Functions:

* check_dataset(dataset_dir): Verify the presence of annotations for each image.
* convert_annotation(dataset_dir, classes): Convert annotations to YOLOv8 format and create a class mapping YAML file.
* get_dataset_plots(dataset_dir, classes): Generate bar plots displaying dataset statistics, including the number of images per country and objects per class for each country.
* visualize_dataset(dataset_dir, num_images): Display random images from the dataset with their bounding boxes and class labels.
* merge_country_datasets(dataset_dir): Combine all country-specific datasets into one "ALL" dataset without altering the original datasets.
* remove_empty_images_and_labels(dataset_dir): Eliminate empty labels (indicating no object presence) and the corresponding images.
* split_dataset(dataset_dir, dataset_statistics): Split the dataset into train/valid/test sets based on class distribution.
* plot_class_distribution(dataset_dir, classes): Print and plot the class distribution after splitting the dataset.
* create_yaml(dataset_dir, classes): Generate a YAML file for the dataset, an essential requirement for YOLOv8.


### Usage:

#### Install required Python libraries:


#### Define"dataset_dir" variable as following:
dataset_dir = path to "RDD2022" folder containing the 7 sub-datasets (countries-datasets) as seen in the dataset structure above.

#### Note:
Some functions should only be excuted once. If excuted more than once then the dataset might be ruined! See comments in the preprocessing.py script.


2. yolov8.py
This Python script facilitates training a YOLOv8 model using the preprocessed RDD2022 dataset. Customization of dataset paths, configuration, and model training settings is required.
It will be added later In Shaa Allah.
