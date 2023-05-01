"""preprocess and prepare the dataset for training"""

import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from configs import Config

import monai
from monai import transforms

def insert_cases_paths_to_df(df:str, train_dir:str = None, test_dir:str = None):
    """insert full cases paths to name mapping dataframe for data loading and data
    preparation
    
    Parameters
    ----------
    df: str
    train_dir: str
    test_dir: str
    
    Return:
    df: pd.DataFrame processed"""
    df = pd.read_csv(df)
    paths = []
    phase = []
    df = df[df['BraTS2021'].notna()]
    for _ , row in df.iterrows():
        id = row["BraTS2021"]
        if id in train_dir:
            path = os.path.join(train_dir, id)
            type = "train"
        elif id in test_dir:
            path = os.path.join(test_dir, id)
            type = "val"
        paths.append(path)
        phase.append(type)
    df['path'] = paths
    df['fold'] = phase
    return df
    

def data_transforms(phase: str = 'train'):
    '''apply data transforms to an 3D image
    
    Parameters
    ----------
    phase: str
    
    Return:
    -------
    transform: transforms.Compose
    '''  
    train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.CropForegroundd(
                    keys=["image", "label"],
                    source_key="image",
                    k_divisible=[Config.newGlobalConfigs.swinUNetCongis.roi[0], 
                                 Config.newGlobalConfigs.swinUNetCongis.roi[1], 
                                 Config.newGlobalConfigs.swinUNetCongis.roi[2]],
                ),
                transforms.RandSpatialCropd(
                    keys=["image", "label"],
                    roi_size=[roi[0], roi[1], roi[2]],
                    random_size=False,
                ),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0)])
                
    val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ])
    transform = {'train': train_transform, 'val': val_transform}
    return transform[phase]