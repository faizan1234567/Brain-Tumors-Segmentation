'''data loader classe, load Brats-20 data for training'''
from configs import Config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset
import nibabel as nib
from skimage.transform import resize
# from segment_3d import ConvertToMultiChannelBasedforBratsClassesd
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType)

from monai.utils import set_determinism
set_determinism(seed=0)

class BratsDataset20(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str="test", is_resize: bool=False):
        self.df = df
        self.phase = phase
        self.augmentations = augmentations(phase)
        self.data_types = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz']
        self.is_resize = is_resize
        # self.val_dir = Config.Testset.dir_path
        # self.test_dir
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        id_ = self.df.loc[idx, 'Brats20ID']
        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]
        # load all modalities
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)
            
            if self.is_resize:
                img = self.resize(img)
    
            img = self.normalize(img)
            images.append(img)
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
        
        # there is no labels in the test directory
        if self.phase != "test":
            mask_path =  os.path.join(root_path, id_ + "_seg.nii.gz")
            mask = self.load_img(mask_path)
            
            if self.is_resize:
                mask = self.resize(mask)
                mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
                mask = np.clip(mask, 0, 1)
            mask = self.preprocess_mask_labels(mask)
            data = {"image": img.astype(np.float32),
                    "mask": mask.astype(np.float32)}
    
            augmented = self.augmentations(data)
            
            img = augmented['image']
            mask = augmented['mask']
            return {
                "image": img,
                "mask": mask}
        
        return {
            "image": img}
    
    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data
    
    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)
    
    def resize(self, data: np.ndarray):
        data = resize(data, (78, 120, 120), preserve_range=True)
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

def augmentations(phase):
  '''apply data augmentation options on 3D medical images, on some cases image
  transformation is needed to make image amenable to the network. List of the transfromation
  and augmentation options are written below, for more please see monai.transfroms
  
  Args:
  phase: string
    training or test transformations to select using this flag
  
  Return:
  train_transform or test_transfom based on phase flag'''

  if phase == 'train':
    train_transform = Compose([
            EnsureTyped(keys=["image", "mask"]),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            Spacingd(
                keys=["image", "mask"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "mask")),

            RandSpatialCropd(keys=["image", "mask"], roi_size=[96, 224, 224], random_size=False),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0)])
    return train_transform

  elif phase == 'val':
    val_transform = Compose([
            EnsureTyped(keys=["image", "mask"]),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            Spacingd(
                keys=["image", "mask"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)])
    return val_transform
  else:
    test_transform = Compose([
            EnsureTyped(keys = "image"),
            Orientationd(keys = "image", axcodes = "RAS"),
            Spacingd(
                keys="image",
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)])
    return test_transform
