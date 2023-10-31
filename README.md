# Real-Time Cracks Detection and Severity Classification System

## Background
Road infrastructure is prone to deterioration, and the presence of cracks poses significant safety risks. Early detection of these road defects is essential to enhance safety, reduce maintenance costs, and eliminate the inefficiencies associated with manual inspections. Innovative solutions, such as Industrial Internet of Things (IIoT) technologies, offer promising prospects for addressing these challenges.


## Dataset
This work utilizes the Road Damage Detection 2022 (RDD2022) dataset. This comprehensive dataset can be accessed via this [link](https://doi.org/10.48550/arXiv.2209.08538), and the data can be downloaded from this [source](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547?file=38030820).

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
```


## Repository Contents:

### 1. preprocessing.py
This script contains a set of functions for preprocessing the RDD2022 dataset, making it ready for machine learning.

#### Functions:

* check_dataset(dataset_dir): Verify the presence of annotations for each image.
* convert_annotation(dataset_dir, classes): Convert annotations to YOLOv8 format.
* get_dataset_plots(dataset_dir, classes): Generate bar plots displaying dataset statistics, including the number of images per country and objects per class for each country.
* visualize_dataset(dataset_dir, num_images): Display random images from the dataset with their bounding boxes and class labels.
* merge_country_datasets(dataset_dir): Combine all country-specific datasets into one "ALL" dataset without altering the original datasets.
* remove_empty_images_and_labels(dataset_dir): Eliminate empty labels (indicating no object presence) and the corresponding images.
* split_dataset(dataset_dir, dataset_statistics): Split the dataset into train/valid/test sets based on class distribution.
* plot_class_distribution(dataset_dir, classes): Print and plot the class distribution after splitting the dataset.
* create_yaml(dataset_dir, classes): Generate a YAML file for the dataset, an essential requirement for YOLOv8.


#### Usage:

##### Install required Python libraries:

```
pip install matplotlib opencv-python-headless seaborn PyYAML
```

##### Define"dataset_dir" variable as following:
dataset_dir = path to "RDD2022" folder containing the 7 sub-datasets (countries-datasets) as seen in the dataset structure above.

##### Note:
Some functions should only be excuted once. If excuted more than once then the dataset might be ruined! See comments in the preprocessing.py script.

#### Output
After Running the preprocessing.py script you will get the following:

- New folder called "ALL" will be created inside the "RDD2022" folder.
	- This folder is the dataset that will be used for training and testing YOLOv8
	- This folder contains three folders: "train", "valid", and "test". Additionally, a yaml file will be created.
	- Inside each one of these three folders there are two folders: "images" and "labels"
	- The labels follow YOLOv8 format
- Plots will be generated for data visuallization including:
	- Number of images per country.
	- Number of objects per class for each country.
	- Combined number of objects per class (considering all contries).
	- Class distribution after splitting the dataset into train/valid/test.

### 2. yolov8.py

This Python script will be used for training a YOLOv8 model using the preprocessed RDD2022 dataset.

#### Functions:

* model.train(): This function is used to train the model
* model.val(): This function is used to validate the model

#### Dependencies:

You need to install ultralytics pacakge as following:

```
pip install ultralytics
```

#### Usage:

* YOLOv8 comes with 5 sizes [nano, small, medium, large, xlarge], to choose one of these sizes you just need to specify the corresponding letter of the desired size as follow:

```
# Choose from ['n', 's', 'm', 'l', 'x']
model = YOLO('yolov8s.pt')  # load a small pretrained model
model = YOLO('yolov8n.pt')  # load a nano pretrained model

```

* If you want to build a model from scratch then you should use this code instead of pretrained:

```
# Choose from ['n', 's', 'm', 'l', 'x']
model = YOLO('yolov8n.yaml')
```

* To see all model hyperparameters and settings, you need to check the default.yaml file.
This can be found in "ultralytics\cfg\default.yaml"


### 3. augmentation.py
This Python script is designed to augment image data and copy label data from a training dataset. It uses the `imgaug` library to apply augmentation techniques to the images, and `shutil` to copy the label files.
The augmentation that will be done are mix of brightness, contrast, and hue.


#### Usage:

1. Ensure you have the required libraries installed. You can install them using the following command:

```
pip install imgaug opencv-python-headless
```

2. Specify the path to train dataset folder at the beginning of the code.

3. Run the code

#### Output:

1. After Running the code you will have a new folder inside the RDD2022 dataset folder called "augmented_train" along with "train", "valid", and "test" folders and the yaml file.
2. Now all you need to do is to modify the train path in the yaml file to point toward this "augmented_train" folder.



