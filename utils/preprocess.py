"""preprocess and prepare the dataset for training"""

import pandas as pd
import numpy as np
import os
import json
import random
import matplotlib.pyplot as plt
import sys
from pathlib import Path

import monai
from monai import transforms

def separate_train_val_ids(json_file: str = None,
                           fold: int = 0, 
                           phase: str = 'training'):
    """
    separte out training ids and 
    validation based on the folder index, i.e In training, there should be 
    about 4 folds and 1 folds should be in validation as there are 5 folds.
    
    Parameters
    ----------
    json_file: str
    fold: int"""

    with open(json_file, 'r') as file:
        data = file.read()
    dataset = json.loads(data)
    dataset = dataset[phase]
    training = []
    validation = []
    for example in dataset:
        if example['fold'] == fold:
            patient_id = example['label'].split('/')[-2]
            validation.append(patient_id)
        else:
            patient_id = example['label'].split('/')[-2]
            training.append(patient_id)

    return (training, validation)
        
def insert_cases_paths_to_df(df:str, 
                             train_dir:str = None, 
                             test_dir:str = None, 
                             json_file:str = None, 
                             fold: int = 0):
    """
    insert full cases paths to name mapping dataframe 
    for data loading and data preparation
    
    Parameters
    ----------
    df: str
    train_dir: str
    test_dir: str
    
    Return:
    df: pd.DataFrame processed"""
    df = pd.read_excel(df)
    paths = []
    phase = []
    train, val = separate_train_val_ids(json_file= json_file, fold= fold)
    df = df[df['BraTS2023'].notna()]
    for _ , row in df.iterrows():
        id = row["BraTS2023"]
        if id in os.listdir(train_dir):
            path = train_dir + "/" + id
            if id in train:
                type = "train"
            elif id in val:
                type = "val"
            else:
                type = None
        elif id in os.listdir(test_dir):
            path = test_dir + "/" + id
            type = "test"
        paths.append(path)
        phase.append(type)
    df['path'] = paths
    df['phase'] = phase
    return df
    

def data_transforms(phase: str = 'train', roi: int = 128):
    '''apply data transforms to an 3D image
    
    Parameters
    ----------
    phase: str
    
    Return:
    -------
    transform: transforms.Compose
    '''  
    train_transform = transforms.Compose(
            [   transforms.EnsureTyped(keys=["image", "label"]),
                transforms.CropForegroundd(
                    keys=["image", "label"],
                    source_key="image",
                    k_divisible=[roi, roi, roi],
                ),
                transforms.RandSpatialCropd(
                    keys=["image", "label"],
                    roi_size=[roi, roi, roi],
                    random_size=False,
                ),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                transforms.ToTensord(keys=["image", "label"])])
                
    val_transform = transforms.Compose(
            [
                transforms.EnsureTyped(keys=["image", "label"]),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image", "label"])
            ])
    test_transform = val_transform
    transform = {'train': train_transform, 'val': val_transform}
    return transform[phase]
