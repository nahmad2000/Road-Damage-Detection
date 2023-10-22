# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 21:07:10 2023

@author: ahmad
"""



#%%     Variables

data = # Provide the full path to the yaml file that was created inside "RDD2022/ALL" after running the preprocessing.py script.

#%%     Some train settings (used in model.train())

# You don't need to change anything here to run the code (but you can change if you wish)

epochs = 3 # Number of epochs for training (int)
patience = 50  # (int) epochs to wait for no observable improvement for early stopping of training
batch = 16  # (int) number of images per batch (-1 for AutoBatch)
imgsz = 640  # (int | list) input images size as int for train and val modes, or list[w,h] for predict and export modes
workers = 8  # (int) number of worker threads for data loading (per RANK if DDP)
pretrained = True  # (bool | str) whether to use a pretrained model (bool) or a model to load weights from (str)
optimizer =  'auto'  # (str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
cos_lr = False  # (bool) use cosine learning rate scheduler
resume = False  # (bool) resume training from last checkpoint


# To see all model settings and hyperparameters check the default.yaml file
# It can be found here: ultralytics\cfg\default.yaml

#%%     Get Yolo mode


# check software and hardware
import ultralytics
ultralytics.checks()
from ultralytics import YOLO


# Load a model ['n', 's', 'm', 'l', 'x']
# model = YOLO('yolov8n.yaml')  # build a new model from scratch
model = YOLO('yolov8s.pt')  # load a pretrained model 


#%%     Train the model
results = model.train(data=data, epochs=epochs)  # train the model

#%%     Validate the model
results = model.val()  # evaluate model performance on the validation set
    

#%%     Test the model (inferance)

image_path = # Path to the image to test the model on

results = model(image_path)  # predict on an image


#%%     Export the model 


# results = model.export(format='onnx')  # export the model to ONNX format
    
    