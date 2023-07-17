'''
BraTS 2023 Data loading
-----------------------
Load the dataset from the disk in batches for training and inference
Author: Muhammad Faizan
Copyright (c) 2022 Muhammad Faizan
'''
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import nibabel as nib
import os
from skimage.transform import resize
from pathlib import Path
import sys
import logging

#setting up loggers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logs/dataset.log")
stream_handler = logging.StreamHandler()

formatter = logging.Formatter(" %(asctime)s: %(name)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)
# logger.info("The logger configured.")

try:
    from config.configs import Config
    from utils.preprocess import data_transforms
    from utils.preprocess import insert_cases_paths_to_df
except ModuleNotFoundError:
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1]  #project root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    from config.configs import Config
    from utils.preprocess import data_transforms
    from utils.preprocess import insert_cases_paths_to_df

class BraTSDataset(Dataset):
    """Load BraTS 2021 dataset from the disk for training and 
    validation and testing as per the requirment
    
    Parameters
    ----------
    df: pd.DataFrame
    phase: str
    is_resize: bool
    """
    def __init__(self, df: pd.DataFrame, 
                 phase: str = "val", 
                 is_resize: bool = False, 
                 is_process_mask: bool = True):
        self.df = df
        self.phase = phase
        self.is_resize = is_resize
        self.augmentations = data_transforms(phase= phase)
        self.data_types = ['-t2f.nii.gz', '-t1n.nii.gz', '-t1c.nii.gz', '-t2w.nii.gz']
        self.is_process_mask = is_process_mask

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        """
        Get the specific dataset sample from the index specified

        Parameters
        ----------
        index: int (ith index into the dataset)

        Returns
        -------
        sample: dict i.e. {"image", "mask"}
        """
        #get root path
        case = self.df.loc[index, "BraTS2023"]
        root_path = self.df.loc[self.df['BraTS2023'] == case]['path'].values[0]
        
        # load and stack modalities
        modalities = [] 
        for data_type in self.data_types:
            img_path = root_path + "/" + case + data_type
            img = self.load_img(img_path)
            
            if self.is_resize:
                img = self.resize(img)
    
            img = self.normalize(img)
            modalities.append(img)
        img = np.stack(modalities)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
        
        # load the label for non test set
        if self.phase != "test":
            mask_path =  root_path + "/" + case + "-seg.nii.gz"
            mask = self.load_img(mask_path)
            mask = np.moveaxis(mask, (0, 1, 2), (2, 1, 0))
            
            # resize to the given dimensions
            if self.is_resize:
                mask = self.resize(mask)
                mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
                mask = np.clip(mask, 0, 1)

            # convert the mask to 3 channels as there are 3 tumors classes
            if self.is_process_mask or self.phase == "training":
                mask = self.preprocess_mask_labels(mask)
                mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

            data = {"image": img.astype(np.float32),
                    "label": mask.astype(np.float32)}
            
            # apply data augmentation and transforms
            augmented = self.augmentations(data)
            
            img = augmented['image']
            mask = augmented['label']
    
            # return labelled sample
            return {
                "image": img,
                "label": mask,
            }
        
        # return test set sample where labels are not available or attentionally not used
        return {
            "image": img,
        }
    
    def load_img(self, file_path: str):
        """
        Load nii.gz format file from the given path
        
        Parameters
        ----------
        file_path: str (specfied path)

        Returns
        -------
        data: np.ndarray (in numpy format)
        """
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data
    
    def normalize(self, data: np.ndarray):
        """
        Normalize an image 

        Parameters
        ----------
        data: np.ndarray 

        Returns
        -------
        data: np.ndarray (normalized)
        """
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    def resize(self, data: np.ndarray):
        """
        Resize the image into required dimensions or size

        Parameters
        ----------
        data: np.ndarray 
         
        Returns 
        -------
        data: np.ndarray (Resized image)
        """
        data = resize(data, (78, 120, 120), preserve_range=True)
        return data
    
    def preprocess_mask_labels(self, mask: np.ndarray):
        """
        Process mask label and creat 3 channels 
        output based on three tumor types

        Parameters
        ----------
        mask: np.ndarray

        Returns
        -------
        mask: np.ndarray (stacked with wt, tc, and et)
        """
        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        return mask


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    path_to_csv: str,
    phase: str,
    batch_size: int = 1,
    num_workers: int = 2,
    json_file: str = None,
    fold: int = 0,
    train_dir: str = Config.newGlobalConfigs.train_root_dir,
    test_dir: str = Config.newGlobalConfigs.test_root_dir,
    is_process_mask: bool = True):

    '''
    Load the dataset into batches from the disk.

    Parameters
    ----------
    dataset: torch.utils.data.Dataset (a datset instance)
    path_to_csv: str (path to a xlsx or csv file)
    phase: str (training, validation, or test phase)
    batch_size: int (loader batch size to be specified)
    num_workers: int (number of workers to be employed to load data)
    json_file: str (path to dataset split json file)
    fold: int (fold index to be specified for validation phase)
    train_dir: str (training directory)
    test_dir: str(test directory)
    is_process_mask: bool (whether to stack tumor classes to the original label)


    Returns:
    -------
    dataloader: torch.utils.data.DataLoader (loader object)
    '''
    # get the dataframe with full path to patients records
    df = insert_cases_paths_to_df(path_to_csv, train_dir = train_dir, 
                                  test_dir= test_dir, json_file=json_file, fold=fold)

    # seperate training data frame from validation data frame
    train_df = df.loc[df['phase'] == 'train'].reset_index(drop=True)
    val_df = df.loc[df['phase'] == 'val'].reset_index(drop=True)
    df = train_df if phase == "train" else val_df
    logger.info(" {} phase selected.".format(phase.capitalize()))
    logger.info(" First row in the data frame: {}".format(df.iloc[0, :].tolist()))
    
    # Load the dataset for the phase specified
    dataset = dataset(df, phase, is_process_mask = is_process_mask)
    logger.info(f' Total examples in the {phase} dataset: {len(dataset)}')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last= True)
    return dataloader

if __name__ == '__main__':
    # call loader here and test everything works well?...
    json_file = Config.newGlobalConfigs.json_file
    data = get_dataloader(dataset=BraTSDataset, 
                          path_to_csv=Config.newGlobalConfigs.path_to_xlsx, 
                          phase='val', batch_size=2, num_workers=2, 
                          json_file=json_file, is_process_mask= True)
    batch = next(iter(data))
    image, label = batch["image"], batch['label']
    logger.info(f' Image size: {image.shape}, label size: {label.shape}')