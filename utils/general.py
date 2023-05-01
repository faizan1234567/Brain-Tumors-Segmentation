"""Some helper functions"""

import torch
import monai
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def save_checkpoint(model, epoch, filename="model.pt", best_acc=0, save_dir=None):
    """save model information and checkpoints in the save dir
    credit: MONAI

    Parameters
    ----------
    model: torch.nn.Module
    epoch: int
    filename: str
    best_acc: int
    save_dri: str
    """
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(save_dir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

def visualize_data_sample(data_dir, slice=78, modality= "flair"):
    """visualize a modality along with the segmentation label in a subplot
    
    Parmeters
    ---------
    data_dir: str
    slice: int
    modality: str"""
    img_add = os.path.join(data_dir, f"TrainingData/BraTS2021_00006/BraTS2021_00006_{modality}.nii.gz")
    label_add = os.path.join(data_dir, "TrainingData/BraTS2021_00006/BraTS2021_00006_seg.nii.gz")
    img = nib.load(img_add).get_fdata()
    label = nib.load(label_add).get_fdata()
    print(f"image shape: {img.shape}, label shape: {label.shape}")
    IMAGES = [img, label]
    TITLES =["Image", "Label"]
    fig, axes = plt.subplots(1, 2, fig_size = (18, 6))
    for i in range(len(IMAGES)):
        axes[i].imshow(IMAGES[i][:, :, slice], cmap = 'gray')
        axes[i].set_title(TITLES[i])
        axes[i].set_axis_off()
    plt.show()

def seed_everything(seed: int):
    """generate a random seed
    
    Parameters:
    ----------
    seed: int
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)