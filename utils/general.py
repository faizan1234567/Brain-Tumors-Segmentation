"""Some helper functions"""

import torch
import monai
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from config.configs import*

def save_checkpoint(model, epoch, filename="model.pt", best_acc=0, save_dir=os.getcwd()):
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

def visualize_data_sample(case, id,  slice=78, modality= "flair"):
    """visualize a modality along with the segmentation label in a subplot
    
    Parmeters
    ---------
    case: str
    slice: int
    modality: str"""
    # img_add = os.path.join(data_dir, f"TrainingData/BraTS2021_00006/BraTS2021_00006_{modality}.nii.gz")
    # label_add = os.path.join(data_dir, "TrainingData/BraTS2021_00006/BraTS2021_00006_g.nii.gz")
    test_image = case + f"/{id}_{modality}.nii.gz"
    label = case + f"/{id}_seg.nii.gz"
    img = nib.load(test_image).get_fdata()
    label = nib.load(label).get_fdata()
    print(f"image shape: {img.shape}, label shape: {label.shape}")
    IMAGES = [img, label]
    TITLES =["Image", "Label"]
    fig, axes = plt.subplots(1, 2, figsize = (18, 6))
    for i in range(len(IMAGES)):
        if i == 0:
            axes[i].imshow(IMAGES[i][:, :, slice], cmap = 'gray')
        else:
            axes[i].imshow(IMAGES[i][:, :, slice])

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

def plot_train_histroy(data):
    """plot training history of the model
    
    Parameters
    ----------
    data_df: dict
    """
    NAMES = ["training_loss", "WT", "ET", "TC", "mean_dice", "epochs"]
    data_lists = []
    for name in NAMES:
        data_list = data[name]
        data_lists.append(data_list)
    
    


if __name__ == "__main__":
    print()
    print('Loading the patient case for visualization')
    visualize_data_sample(Config.newGlobalConfigs.full_patient_path, Config.newGlobalConfigs.a_test_patient)