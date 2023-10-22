# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 12:47:14 2023

@author: ahmad
"""

# Importation

import matplotlib.pyplot as plt
import os 
import xml.etree.ElementTree as ET
import seaborn as sns
import random
import cv2
import shutil
from collections import defaultdict
import yaml


#%%     Variables 
dataset_dir = r'D:\Ahmad\Work\KFUPM\Term 231\Senior Project\Computer Vision\Datasets\RDD2022\RDD2022'

classes = ['D00', 'D01', 'D10', 'D11', 'D20', 'D40', 'D43', 'D44']



#%%     Functions

def check_dataset(dataset_dir):
    '''
    This function will check whether each image has an annotation or not.
    If an image does not have an annotation then a print statement will print the path of that image.
    
    It takes:
        - dataset_dir (str): path to the dataset folder.
    '''

    # List of country folders in the dataset
    country_folders = os.listdir(dataset_dir)

    for country in country_folders:

        train_folder = os.path.join(dataset_dir, country, "train")
        images_folder = os.path.join(train_folder, "images")
        annotations_folder = os.path.join(train_folder, "annotations", "xmls")

        # Get a list of image files
        image_files = os.listdir(images_folder)

        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            annotation_file = os.path.join(annotations_folder, os.path.splitext(image_file)[0] + ".xml")

            if not os.path.exists(annotation_file):
                print(f"Image {image_path} does not have a corresponding annotation.")





def convert_annotation(dataset_dir, classes):
    '''
    This function will convert the annotation format from the original format (similar to pascal voc) to YOLOv8 format.
    This will create a new "labels" folder for each "annotations" folder and a YAML file for class mapping.
        
    It takes:
        - dataset_dir (str): path to the dataset folder.
        - classes (list): list of strings containing all classes.
    '''
    # Iterate through country folders
    country_folders = os.listdir(dataset_dir)
    for country in country_folders:

        train_folder = os.path.join(dataset_dir, country, "train")
        annotations_folder = os.path.join(train_folder, "annotations", "xmls")

        # Create a "labels" folder for YOLOv8 format
        labels_folder = os.path.join(train_folder, "annotations", "labels")
        os.makedirs(labels_folder, exist_ok=True)

        # Create a YAML file for class mapping
        yaml_file = os.path.join(train_folder, "annotations", "class_mapping.yaml")
        with open(yaml_file, "w") as f:
            for i, class_name in enumerate(classes):
                f.write(f"{i}: {class_name}\n")

        # Iterate through XML files and convert to YOLO format
        for xml_file in os.listdir(annotations_folder):
            if xml_file.endswith(".xml"):
                tree = ET.parse(os.path.join(annotations_folder, xml_file))
                root = tree.getroot()

                image_width = int(root.find(".//width").text)
                image_height = int(root.find(".//height").text)

                yolo_lines = []
                for obj in root.findall(".//object"):
                    class_name = obj.find("name").text
                    if class_name in classes:
                        class_index = classes.index(class_name)

                        
                        # Ensure coordinates are integers and within valid range
                        xmin = min(int(round(float(obj.find(".//xmin").text))), image_width - 1)
                        ymin = min(int(round(float(obj.find(".//ymin").text))), image_height - 1)
                        xmax = min(int(round(float(obj.find(".//xmax").text))), image_width - 1)
                        ymax = min(int(round(float(obj.find(".//ymax").text))), image_height - 1)


                        # Normalize coordinates
                        x_center = (xmin + xmax) / (2.0 * image_width)
                        y_center = (ymin + ymax) / (2.0 * image_height)
                        width = (xmax - xmin) / image_width
                        height = (ymax - ymin) / image_height

                        yolo_lines.append(f"{class_index} {x_center} {y_center} {width} {height}")

                # Write YOLO format annotation to a .txt file
                output_txt_file = os.path.splitext(xml_file)[0] + ".txt"
                output_txt_path = os.path.join(labels_folder, output_txt_file)
                with open(output_txt_path, "w") as f:
                    f.write("\n".join(yolo_lines))





def get_dataset_plots(dataset_dir, classes):
    '''
    This function will generates three bar plots showing statistics related to datasets. The plots are as follow:
    1. Number of images per country.
    2. Number of objects (instances) per class for each country.
    3. Number of objects (instances) per class for the combined result of all countries.

    It takes:
        - dataset_dir (str): path to the dataset folder.
        - classes (list): list of strings containing all classes.
    '''

    # Initialize variables to store overall statistics
    total_images = 0
    total_instances = 0

    # Lists to store data for plotting
    countries = []
    num_images = []
    class_instances = defaultdict(lambda: defaultdict(int))  # Dictionary to store class instances per country

    # Iterate through country folders
    country_folders = os.listdir(dataset_dir)
    for country in country_folders:
        
        if country == 'ALL':
            continue  # skip the "ALL" folders (just in case you run this function after combining datasets into one dataset called "ALL")

        train_folder = os.path.join(dataset_dir, country, "train")
        labels_folder = os.path.join(train_folder, "annotations", "labels")

        # Initialize variables to store country-specific statistics
        country_images = 0
        country_instances = 0

        # Iterate through label files and count instances
        label_files = os.listdir(labels_folder)
        for label_file in label_files:
            with open(os.path.join(labels_folder, label_file), 'r') as f:
                lines = f.readlines()
                country_instances += len(lines)

                # Count class instances per country
                for line in lines:
                    class_id = int(line.split()[0])
                    class_instances[country][classes[class_id]] += 1
        
        # Count the number of image files in the "images" folder
        images_folder = os.path.join(train_folder, "images")
        country_images = len([f for f in os.listdir(images_folder) if f.endswith('.jpg')])

        # Update overall statistics
        total_images += country_images
        total_instances += country_instances

        # Add data for plotting
        countries.append(country)
        num_images.append(country_images)



    # Plot 1: Number of images per country
    sorted_data = sorted(zip(num_images, countries), reverse=True)
    sorted_num_images, sorted_countries = zip(*sorted_data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(sorted_num_images), y=list(sorted_countries), palette="viridis")
    plt.title("Number of Images per Country (Ranked)")
    plt.xlabel("Number of Images")
    plt.ylabel("Country")
    plt.show()

    # Plot 2: Number of objects (instances) per class for each country
    plt.figure(figsize=(15, 15))  # Adjust the figsize to fit all subplots
    plt.suptitle("Number of Objects (Instances) per Class", fontsize=16)
    
    for i, country in enumerate(countries):
        class_counts = class_instances[country]
        sorted_data = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_class_names, sorted_class_counts = zip(*sorted_data)
    
        plt.subplot(3, 3, i + 1)
        sns.barplot(x=list(sorted_class_counts), y=list(sorted_class_names), palette="magma")
        plt.title(country, fontsize=12)  # Add title for each subplot
        plt.xlabel("Number of Objects (Instances)")
        plt.ylabel("Class")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust subplot layout to make space for the main title
    plt.show()


    # Plot 3: Number of objects (instances) per class for the combined result of all countries
    combined_class_instances = defaultdict(int)
    for country, counts in class_instances.items():
        for class_name, count in counts.items():
            combined_class_instances[class_name] += count

    # Sort the data by the number of objects (instances) per class in descending order
    sorted_data = sorted(combined_class_instances.items(), key=lambda x: x[1], reverse=True)
    sorted_class_names, sorted_class_counts = zip(*sorted_data)
    
    plt.figure(figsize=(12, 6))
    plt.bar(list(sorted_class_names), list(sorted_class_counts), color="c")
    plt.title("Combined Number of Objects (Instances) per Class (Ranked)")
    plt.xlabel("Class")
    plt.ylabel("Number of Objects (Instances)")
    plt.xticks(rotation=45, ha="right")
    plt.show()

    return combined_class_instances


def visualize_dataset(dataset_dir, num_images):
    '''
    This function will show random images from the dataset with their bounding boxes and class labels.
    
    It takes:
       - dataset_dir (str): path to the dataset folder. 
       - num_images (int): number of random images to visualize.
    '''

    # Create a list of image paths and corresponding annotation paths
    image_paths = []
    annotation_paths = []
    
    # Iterate through country folders
    country_folders = os.listdir(dataset_dir)
    for country in country_folders:
        if country == "China_Drone":
            train_folder = os.path.join(dataset_dir, country, "train")
            images_folder = os.path.join(train_folder, "images")
            annotations_folder = os.path.join(train_folder, "annotations", "labels")
        else:
            train_folder = os.path.join(dataset_dir, country, "train")
            images_folder = os.path.join(train_folder, "images")
            annotations_folder = os.path.join(train_folder, "annotations", "labels")

        image_files = os.listdir(images_folder)
        for image_file in image_files:
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(images_folder, image_file))
                annotation_paths.append(os.path.join(annotations_folder, os.path.splitext(image_file)[0] + ".txt"))

    if len(image_paths) == 0:
        print("No images found in the dataset.")
        return

    # Randomly select images to visualize
    random.seed(0)  # Set a seed for reproducibility
    selected_indices = random.sample(range(len(image_paths)), min(num_images, len(image_paths)))

    # Visualize selected images with bounding boxes and labels
    for i, idx in enumerate(selected_indices):
        image_path = image_paths[idx]
        annotation_path = annotation_paths[idx]

        image = cv2.imread(image_path)
        h, w, _ = image.shape

        # Read YOLO format annotations
        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            class_index, x_center, y_center, width, height = map(float, line.strip().split())
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            class_label = classes[int(class_index)]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Image {i + 1}")
        plt.show()

        
def merge_country_datasets(dataset_dir):
    '''
    This function will merge all the country datasets into one big dataset without modifying the original datasets.
    The merged dataset "ALL" will be created in the same path of the dataset.

    It takes:
       - dataset_dir (str): path to the folder containing country datasets.
    '''

    # Iterate through country datasets
    country_folders = os.listdir(dataset_dir)
    for country in country_folders:
        country_dataset_dir = os.path.join(dataset_dir, country, "train")
        country_images_dir = os.path.join(country_dataset_dir, "images")
        country_labels_dir = os.path.join(country_dataset_dir, "annotations", "labels")

        # Check if the country dataset exists
        if not os.path.exists(country_dataset_dir):
            print(f"Skipping '{country}' dataset - folder not found.")
            continue

        print(f"Merging '{country}' dataset...")

        # Create the "ALL" dataset folder structure on the first iteration
        if country == country_folders[0]:
            all_dataset_dir = os.path.join(dataset_dir, "ALL")
            all_train_dir = os.path.join(all_dataset_dir, "train")
            all_images_dir = os.path.join(all_train_dir, "images")
            all_labels_dir = os.path.join(all_train_dir, "labels")

            # Create necessary directories if they don't exist
            os.makedirs(all_images_dir, exist_ok=True)
            os.makedirs(all_labels_dir, exist_ok=True)

        # Copy images from the country dataset to the "ALL" dataset
        for image_file in os.listdir(country_images_dir):
            source_image_path = os.path.join(country_images_dir, image_file)
            destination_image_path = os.path.join(all_images_dir, image_file)
            shutil.copy(source_image_path, destination_image_path)

        # Copy annotations from the country dataset to the "ALL" dataset
        for annotation_file in os.listdir(country_labels_dir):
            source_annotation_path = os.path.join(country_labels_dir, annotation_file)
            destination_annotation_path = os.path.join(all_labels_dir, annotation_file)
            shutil.copy(source_annotation_path, destination_annotation_path)

    print("All country datasets have been merged into the 'ALL' dataset.")




def remove_empty_images_and_labels(dataset_dir):
    '''
    This function will go throught all dataset labels and remove every empty label indicating no object present in the image
    Additionally, the corresponding image will be deleted.
    It takes:
        - dataset_dir (str): path to the folder containing country datasets.
    '''
    labels_dir = os.path.join(dataset_dir, "all", "train", "labels")
    images_dir = os.path.join(dataset_dir, "all", "train", "images")

    empty_images_count = 0

    # Iterate through label files
    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        image_path = os.path.join(images_dir, label_file.replace(".txt", ".jpg"))

        try:
            # Check if the label file is empty
            with open(label_path, 'r') as file:
                if not file.read():
                    empty_images_count += 1
                    # Delete the empty label file
                    os.remove(label_path)
                    # Delete the corresponding image
                    os.remove(image_path)
        except PermissionError as e:
            # Close the file if it is open and then remove it
            try:
                file.close()
            except:
                pass
            try:
                os.remove(label_path)
                os.remove(image_path)
            except Exception as e:
                print(f"PermissionError: {e} (Skipped {label_path} and {image_path})")

    print(f"Number of images with no labels: {empty_images_count}")
    print("Empty label files and their corresponding images have been removed.")



def split_dataset(dataset_dir, dataset_statistics):
    
    '''
    This function will split the dataset into train/valid/test with a split ratio 0.7/0.1/0.2
    
    It takes:
        - dataset_dir (str): path to the dataset folder. 
        - dataset_statistics (dict): dictionary containing the number of objects per class.
        
    The combined_dataset_dir contains only "train" folder, so after runing this function:
        We will have 2 more folders: "valid" and "test"
    
    This function will take into account not only the number of images in train/valid/test. It will also
    take into account the number of objects per class in train/valid/test.
    
    Approach to achieve this:
        1) rearrange the dataset_statistics dict to have classes ranked from min to max
        2) iterate thorugh keys of dataset_statistics dict class by class
        3) for class_name in dataset_statistics.keys():
            3.1) create empty files_list list
            3.2) itertate thoruh all labels
            3.3) if a label contains an object from this class_name, then add this label file to files_list
            3.4) if len(files_list) = dataset_statistics[class_name]: stop itertating through labels
            3.5) else: keep iterating unitl you finish all labels
            3.6) randomly shuffle the files_list
            3.7) move 10% of files_list into "valid" folder
            3.8) move 20% of files_list into "test" folder
            3.9) 70% of files_list should remiain in same path "train" folder
            3.10) clear files_list
    '''
    
    # Define the split ratios
    train_ratio = 0.7
    valid_ratio = 0.1
  # test_ratio = 0.2
    
    # Create the train/valid/test folders if they don't exist
    for folder in ["valid", "test"]:
        for sub_folder in ["images", "labels"]:
            folder_path = os.path.join(dataset_dir, 'all', folder, sub_folder)
            os.makedirs(folder_path, exist_ok=True)
    
    # Rearrange the dataset_statistics dict to have classes ranked from min to max
    sorted_classes = sorted(dataset_statistics.keys(), key=lambda k: dataset_statistics[k])
    
    for class_name in sorted_classes:
        # Get the number of label files for this class
        num_files = dataset_statistics[class_name]
        
        # Create an empty list to collect label files for this class
        files_list = []
        
        labels_dir = os.path.join(dataset_dir, "all", "train", "labels")
        images_dir = os.path.join(dataset_dir, "all", "train", "images")
        
        for label_file in os.listdir(labels_dir):
            label_path = os.path.join(labels_dir, label_file)
            
            with open(label_path, 'r') as file:
                lines = file.readlines()
                # Check if this label file contains objects from the current class
                if any(int(line.split()[0]) == sorted_classes.index(class_name) for line in lines):
                    files_list.append(label_file)
                    if len(files_list) == num_files:
                        break

        
        # Randomly shuffle the list
        random.shuffle(files_list)
        
        # Determine the split sizes
        num_train = int(train_ratio * num_files)
        num_valid = int(valid_ratio * num_files)
        
        # Split the files_list into train, valid, and test
        valid_files = files_list[num_train:num_train + num_valid]
        test_files = files_list[num_train + num_valid:]
        
        # Move files to the corresponding folders
        for file in valid_files:
            source_label_path = os.path.join(labels_dir, file)
            source_image_path = os.path.join(images_dir, file.replace(".txt", ".jpg"))
            dest_label_path = os.path.join(dataset_dir, "all", "valid", "labels", file)
            dest_image_path = os.path.join(dataset_dir, "all", "valid", "images", file.replace(".txt", ".jpg"))
            
            shutil.move(source_label_path, dest_label_path)
            shutil.move(source_image_path, dest_image_path)
        
        for file in test_files:
            source_label_path = os.path.join(labels_dir, file)
            source_image_path = os.path.join(images_dir, file.replace(".txt", ".jpg"))
            dest_label_path = os.path.join(dataset_dir, "all", "test", "labels", file)
            dest_image_path = os.path.join(dataset_dir, "all", "test", "images", file.replace(".txt", ".jpg"))
            
            shutil.move(source_label_path, dest_label_path)
            shutil.move(source_image_path, dest_image_path)
    
    print("Dataset has been split into train/valid/test.")
    


def plot_class_distribution(dataset_dir, classes):
    '''
    This function will print and plot the class distribution after we split dataset into train, valid, test
    '''
    
    dataset_split = ["train", "valid", "test"]
    class_counts = {split: {class_name: 0 for class_name in classes} for split in dataset_split}
    
    for split in dataset_split:
        for class_name in classes:
            labels_dir = os.path.join(dataset_dir, "all", split, "labels")
            for label_file in os.listdir(labels_dir):
                label_path = os.path.join(labels_dir, label_file)
                
                with open(label_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        if int(line.split()[0]) == classes.index(class_name):
                            class_counts[split][class_name] += 1
    
    # Print the results
    for split in dataset_split:
        print(f"Class distribution in {split} split:")
        for class_name in classes:
            print(f"{class_name}: {class_counts[split][class_name]} objects")
        print()  # Add an empty line for separation
    
    # Create a bar plot
    x = range(len(classes))
    width = 0.2
    for split in dataset_split:
        counts = [class_counts[split][class_name] for class_name in classes]
        plt.bar([i + width * dataset_split.index(split) for i in x], counts, width, label=split)
    
    plt.xlabel('Classes')
    plt.ylabel('Number of Objects')
    plt.title('Class Distribution in Train, Valid, and Test Splits')
    plt.xticks([i + width for i in x], classes)
    plt.legend()
    
    plt.show()




def create_yaml(dataset_dir, classes):
    '''
    This function will create a yaml file for a given dataset.
    The created yaml file will be in the same given dataset path.
    
    It takes:
        - dataset_dir (str): path to the dataset folder. 
        - classes (list): list of strings containing all classes.
        
    '''
    
    combined_dataset_dir = os.path.join(dataset_dir, 'all')
    data = {
        'train': f'{os.path.join(combined_dataset_dir, "train", "images")}',
        'val': f'{os.path.join(combined_dataset_dir, "valid", "images")}',
        'nc': len(classes),
        'names': classes
    }

    yaml_content = yaml.dump(data, default_flow_style=False)

    yaml_file_path = os.path.join(combined_dataset_dir, 'data.yaml')
    
    with open(yaml_file_path, 'w') as yaml_file:
        yaml_file.write(yaml_content)


                            
                            
                            
#%%     Dataset Preprocessing (check_dataset)

check_dataset(dataset_dir)


#%%     Dataset Preprocessing (convert_annotation)

# **************(This code should be run only once)**************
# **************(This code should be run only once)**************
# **************(This code should be run only once)**************

# convert_annotation(dataset_dir, classes) # Uncomment this line to do the conversion 


#%%     Dataset Preprocessing (get_dataset_plots)

dataset_statistics = get_dataset_plots(dataset_dir, classes)

#%%     Dataset Preprocessing (visualize_dataset)

visualize_dataset(dataset_dir, num_images=10)


#%%     Dataset Preprocessing (merge_country_datasets) 

# **************(This code should be run only once)**************
# **************(This code should be run only once)**************
# **************(This code should be run only once)**************

# merge_country_datasets(dataset_dir) # Uncomment this line to do the merge 


#%%     Dataset Preprocessing (remove_empty_images_and_labels)  ||  (It is enough to run this code only once)

# remove_empty_images_and_labels(dataset_dir) # Uncomment this line to do the removal 


#%%     Dataset Preprocessing (split dataset)

# **************(This code should be run only once)**************
# **************(This code should be run only once)**************
# **************(This code should be run only once)**************

# split_dataset(dataset_dir, dataset_statistics) # Uncomment this line to do the split


#%%     Dataset Preprocessing (plot_class_distribution)

plot_class_distribution(dataset_dir, classes) 


#%%     Dataset Preprocessing (create yaml file) ||  (It is enough to run this code only once)

# create_yaml(dataset_dir, classes) # Uncomment this line to create yaml file





#%%     Some train settings (used in model.train())

data = 'data.yaml' # (str, optional) path to data file, i.e. coco128.yaml
epochs = 3 # Number of epochs for training (int)
patience = 50  # (int) epochs to wait for no observable improvement for early stopping of training
batch = 16  # (int) number of images per batch (-1 for AutoBatch)
imgsz = 640  # (int | list) input images size as int for train and val modes, or list[w,h] for predict and export modes
workers = 8  # (int) number of worker threads for data loading (per RANK if DDP)
pretrained = True  # (bool | str) whether to use a pretrained model (bool) or a model to load weights from (str)
optimizer =  'auto'  # (str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
cos_lr = False  # (bool) use cosine learning rate scheduler
resume = False  # (bool) resume training from last checkpoint


# To see all model settings and hyperparameters can be found here:
# C:\Users\ahmad\anaconda3\envs\YOLOv8\Lib\site-packages\ultralytics\cfg\default.yaml

#%%     Get Yolo mode

# The ultralytics package is avaliable at: C:\Users\ahmad\anaconda3\envs\YOLOv8\Lib\site-packages\ultralytics


# check software and hardware
import ultralytics
ultralytics.checks()
from ultralytics import YOLO


# Load a model ['n', 's', 'm', 'l', 'x']
# model = YOLO('yolov8n.yaml')  # build a new model from scratch
model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)


#%%     Train the model
results = model.train(data=data, epochs=epochs)  # train the model

#%%     Validate the model
results = model.val()  # evaluate model performance on the validation set
    

#%%     Export the model
# results = model('image_path')  # predict on an image
# results = model.export(format='onnx')  # export the model to ONNX format
    
    
    
    
    
    
    
    