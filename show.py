"""
==============================================================
script to show the image and label, or image with labeled mask
==============================================================

Author: Muhammad Faizan
Date: 13 May 2023
Copywrite (c) Muhammad Faizan
==============================================================
"""
import matplotlib.pyplot as plt
import logging
import argparse
import numpy as np
import hydra
from omegaconf import DictConfig
import sys
import nibabel as nib
import os
from pathlib import Path
import cv2
import random

import torch
from brats import get_datasets
from utils.visualizer import visualize_abnormal_area, get_labelled_image, visualize_data_gif
from utils.general import visualize_data_sample
from monai.inferers import sliding_window_inference

from monai.networks.nets import SwinUNETR
from monai.transforms import (
    AsDiscrete,
    Activations,
)

from monai.networks.nets import SwinUNETR, SegResNet, VNet, BasicUNetPlusPlus, AttentionUnet, DynUNet, UNETR
from networks.models.ResUNetpp.model import ResUnetPlusPlus
from networks.models.UNet.model import UNet3D
from networks.models.UX_Net.network_backbone import UXNET
from networks.models.nnformer.nnFormer_tumor import nnFormer
try:
    from thesis.models.SegUXNet.model import SegUXNet
except ModuleNotFoundError:
    print('model not available, please train with other models')
    # sys.exit(1)

from functools import partial

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s: %(name)s: %(message)s")
file_handler = logging.FileHandler(filename= "logger/show.log")
stream_handler = logging.StreamHandler()
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def load_patient_case(path, type = "T1", slice = 75, load_label = True, 
                      plot = True, crop=False):
    modalities = ["t1", "t1ce", "flair", "t2", "seg"]
    if type.lower() not in modalities:
        raise NotImplementedError("The modality type not supported")
    # load nifti image
    patient_id = path.split("/")[-1]
    file_path = Path(path) / f"{patient_id}_{type.lower()}.nii.gz"
    file = nib.load(file_path)
    image = file.get_fdata()
    if load_label:
        label_path = Path(path) / f"{patient_id}_seg.nii.gz"
        label = nib.load(label_path)
        label_image = label.get_fdata()
        label_slice = label_image[:, :, slice]
        label_slice = np.rot90(label_slice, k=-1)

    image_slice = image[:, :, slice]
    image_slice = np.rot90(image_slice, k=-1)
    if crop:
        image_slice = image_slice[48:208, 54:184]
        label_slice = label_slice[48:208, 54:184]

    if plot:
        images = [image_slice, label_slice]
        titles = [f"{type}", "ground-truth"]
        fig, axes = plt.subplots(nrows = 1, ncols =2, figsize = (10, 5))
        for i in range(len(images)):
            if i == 0:
                axes[i].imshow(images[i], cmap= 'gray')
            else:
                axes[i].imshow(images[i])
            axes[i].set_title(titles[i])
            axes[i].set_axis_off()
        plt.show()
    return image_slice, label_slice

def overlay_mask(path, slice=75, type = "T1ce", save_path="media/results", 
                 img_name="ground_truth"):
    os.makedirs(save_path, exist_ok=True)
    image_slice, label_slice = load_patient_case(path=path, type=type, slice=slice, 
                                                 plot=False, crop=True, load_label=True)
    label_slice = label_slice.astype(np.int64)
    image_slice = np.uint8((image_slice - image_slice.min()) / (image_slice.max() - image_slice.min()) * 255)
    overlay_image = np.stack([image_slice] * 3, axis=-1)
    original_image = np.copy(overlay_image)
    
    overlay_image[label_slice == 1] = [255, 0, 0]  # Red
    overlay_image[label_slice == 2] = [0, 255, 0]  # Green
    overlay_image[label_slice == 3] = [0, 0, 255]  # Blue
    if save_path:
        file_path = os.path.join(save_path, img_name + ".png")
        original_img = os.path.join(save_path, f'original_img{slice}_{type}.png')
        cv2.imwrite(file_path, overlay_image)
        cv2.imwrite(original_img, original_image)

    # Display the overlay image
    plt.imshow(overlay_image)
    plt.title('Overlayed Segmentation Mask on Image Slice')
    plt.axis('off')
    plt.show()


def save_slice_75_with_modality_name(patient_path, output_dir):
    """
    Extracts slice 75 from each 3D NIfTI file in the patient's data directory and saves it as a PNG image,
    using only the modality name extracted from the file name.

    Args:
        patient_path (str): Path to the patient's data folder (contains MRI modalities and segmentation mask files).
        output_dir (str): Directory to save the extracted slice images as PNG.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    slice_index = 75

    # Iterate through all NIfTI files in the directory
    for file_name in os.listdir(patient_path):
        if file_name.endswith(".nii") or file_name.endswith(".nii.gz"):  # Process only NIfTI files
            file_path = os.path.join(patient_path, file_name)

            # Extract the modality part of the name (e.g., "t2w" from "BraTS-GLI-00008-001-t2w.nii.gz")
            modality_name = file_name.split('-')[-1].split('.')[0]

            # Load the modality
            img = nib.load(file_path)
            data = img.get_fdata()
            data = np.rot90(data, k=-1)
            # Check if slice 75 exists in the data
            if slice_index < data.shape[2]:
                slice_data = data[:, :, slice_index]

                # Normalize the slice for better visualization (optional)
                slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data) + 1e-8) * 255.0
                slice_data = slice_data.astype(np.uint8)
                slice_data = slice_data[40:220, 50:190]
        
                # Save as PNG with the modality names
                output_path = os.path.join(output_dir, f"{modality_name}_slice_{slice_index}.png")
                plt.imsave(output_path, slice_data, cmap='gray')
                print(f"Saved slice 75 of {modality_name} to {output_path}")
            else:
                print(f"Slice {slice_index} does not exist in {file_name}")


@hydra.main(config_name='configs', config_path= 'conf', version_base=None)
def show_result(cfg: DictConfig, args:argparse.Namespace):
    """
    Visualize labelled brain scan on a patient case, three options are available
    1 - create brain scan slices and label them
    2 - create a .gif format file to visualize part of brain (labelled)
    3 - visualize a scan with it's label in a subplot format
    """
   
    # Load data
    dataset = get_datasets(cfg.dataset.dataset_folder, "test")
    data_loader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=1, 
                                            shuffle=False, num_workers=8, 
                                            pin_memory=True) 
    
    # batch of data
    batch = next(iter(data_loader))
    image, label = batch["image"], batch['label']
    logger.info('visualizing an image with label')

    # Visualize 
    if args.get_abnormal_area:
        visualize_abnormal_area(image, label)
    elif args.visualize_data_gif:
        labelled_img = get_labelled_image(image, label)
        visualize_data_gif(labelled_img)
    elif args.visualize_data_sample:
        visualize_data_sample(args.test_patient_path, 
                              cfg.paths.test_patient)
    else:
        logger.info('No option selected')
        sys.exit()



if __name__ == "__main__":
  
    # Visualize
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_path", default= "media/Brats18_2013_21_1", type = str, 
                        help = "path to patient MRI scan")
    parser.add_argument("--modality", default= "T1ce", type = str, 
                        help = "type of modality type for analysis and visualization")
    
    args = parser.parse_args()
    # show_result(args)
    overlay_mask(path=args.scan_path, slice=75, type="T1", 
                 save_path="media/qualitative_results", img_name="ground_truth")
    