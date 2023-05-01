'''BraTS 2021 Data loader class'''
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import nibabel as nib
import cv2
import os
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  #project root directory
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from configs import Config
from utils.preprocess import data_transforms
from utils.preprocess import insert_cases_paths_to_df

class BraTSDataset(Dataset):
    """Load BraTS 2021 dataset from the disk for training and validation and testing
    as per the requirment
    
    Parameters
    ----------
    df: pd.DataFrame
    phase: str
    is_resize: bool
    """
    def __init__(self, df: pd.DataFrame, phase: str = "val", is_resize: bool = False):
        super(self, BraTSDataset).__init__()
        self.df = df
        self.phase = phase
        self.is_resize = is_resize
        self.augmentations = data_transforms(phase= phase)
        self.data_types = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz']

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        case = self.df.loc[index, "BraTS2021"]
        root_path = self.df.loc[self.df['Brats20ID'] == case]['path'].values

        modalities = []
        for data_type in self.data_types:
            img_path = os.path.join(root_path, case + data_type)
            img = self.load_img(img_path)
            
            if self.is_resize:
                img = self.resize(img)
    
            img = self.normalize(img)
            modalities.append(img)
        img = np.stack(modalities)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
        
        if self.phase != "val":
            mask_path =  os.path.join(root_path, case + "_seg.nii")
            mask = self.load_img(mask_path)
            
            if self.is_resize:
                mask = self.resize(mask)
                mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
                mask = np.clip(mask, 0, 1)
            mask = self.preprocess_mask_labels(mask)
    
            augmented = self.augmentations(image=img.astype(np.float32), 
                                           mask=mask.astype(np.float32))
            
            img = augmented['image']
            mask = augmented['mask']
    
        
            return {
                "image": img,
                "mask": mask,
            }
        
        return {
            "image": img,
        }





    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data
    
    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)
    
    def resize(self, data: np.ndarray):
        data = cv2.resize(data, (78, 120, 120), preserve_range=True)
        return data
    
    def preprocess_mask_labels(self, mask: np.ndarray):

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
    fold: str = 'val',
    batch_size: int = 1,
    num_workers: int = 4,):

    '''Returns: dataloader for the model training'''
    df = pd.read_csv(path_to_csv)
    df = insert_cases_paths_to_df(df, train_dir= Config.newGlobalConfigs.train_root_dir, 
                                  test_dir= Config.newGlobalConfigs.test_root_dir)
    
    train_df = df.loc[df['phase'] == 'train'].reset_index(drop=True)
    val_df = df.loc[df['phase'] == 'val'].reset_index(drop=True)

    df = train_df if phase == "train" else val_df
    dataset = dataset(df, phase)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,   
    )
    return dataloader

if __name__ == '__main__':
    data = get_dataloader(BraTSDataset, Config.newGlobalConfigs.path_to_csv, 'train', "")
    print(iter(next(data)))