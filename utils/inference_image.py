'''Inference on 3D brain tumor image
Author: Muhammad Faizan
The model will take an image and the code will show the mask and true labels on the image
'''

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from segment_3d import inference
import argparse
import nibabel as nib
from configs import Config
from BratsCustom import BratsDataset20
from utils import util
from segment_3d import model_loss_optim
from utils import plot_image_grid
from scripts.tes import categorical

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default= Config.a_test_patient, type = str, help = "path to the case directory in the test/val/train dir")
    parser.add_argument("--weights", default= " ", type = str, help= "weight file path")
    opt = parser.parse_args()
    return opt



def load_image(patient_path, mask_label = True):
    """load the image from the given path
    
    Args:
    patient_path: os.path (str)"""
    dataset = BratsDataset20(phase="val", patient=patient_path, mask_label=mask_label)
    img_data = dataset.__getitem__(1)
    scan = img_data["image"]
    mask = img_data["mask"]
    return (scan, mask)

def show_labeled_image(patient):
    """get labled image for three different orientation such as 
    coronal, transversaral, and sigital for showing enhanced, non-enhaned and edema voxels labels
    This function will examine t he labled image, and those with AI prediction.
    
    Args:
    patient: os.path(str) -> path of patient files directory
    """
    scan, mask = load_image(patient)
    scan = np.einsum('ijkl->klji', scan)
    # mask = np.einsum('jkl->jkl', mask)
    print(mask)
    # mask = mask[:, :, :, 0]
    print("Shape of the scan: {}, and shape of the mask: {}".format(scan.shape, mask.shape))
    image_input = np.random.rand(240, 240, 155, 4)
    true_label = np.random.rand(240, 240, 155)
    image = categorical(scan, mask)
    plot_image_grid(image)
    plt.show()





def diagnose(model, weights, patient, device, 
                  post_trans):
    '''
    Inference on a patient case, in this a trained deep learning model has been used for 
    diagnosing and predicting abnormal area of the brain.

    Args:
    model: torch.nn (a deep learning model)
    weigts: os.path (str) --> model's weights
    patient: os.path (str) --> path to patient directory in the dataset
    device: int (if there is GPU or cpu)
    post_trans: post processing
    '''
    # device = 'cpu'
    model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
    # torch.load(os.path.join(weights, "best_metric_model.pth")))
    
    model.eval()
    with torch.no_grad():
    # select one image to evaluate and visualize the model output
        # val_input = patient[6]["image"].unsqueeze(0).to(device)
        print("loading patient file...")
        val_input, mask = load_image(patient)
        val_input = np.einsum('ijkl->iklj', val_input)
        mask = np.einsum('ijkl->iklj', mask)
        val_input_tensor, mask_tensor = torch.from_numpy(val_input).unsqueeze(0).to(device), torch.from_numpy(mask).to(device).cpu()
        print("Success: the patient file loaded.")
        print("Shape of the scan: {}, and shape of mask: {}".format(val_input.shape, mask.shape))
        
        roi_size = (128, 128, 64)
        sw_batch_size = 1
        val_output = inference(val_input_tensor, model)
        val_output = post_trans(val_output[0]).cpu().numpy()
        # print("output shape: {}".format(val_output.shape))
        plt.figure("Modalities", (24, 6))
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.title(f"Modality {i}")
            plt.imshow(val_input[i, :, :, 75], cmap="gray")
        plt.savefig('patient_modalities_image.png')
        plt.show()

        # visualize the 3 channels label corresponding to this image
        plt.figure("Mask (Ground truth)", (24, 6))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(f"mask {i}")
            plt.imshow(mask_tensor[i, :, :, 70])
        plt.savefig('mask_labels.png')
        plt.show()

        # visualize the 3 channels model output corresponding to this image
        plt.figure("Model Prediction", (24, 6))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(f"prediction {i}")
            plt.imshow(val_output[i, :, :, 70])
        plt.savefig('model_prediction_masks.png')
        plt.show()

def main():
    '''all the code goes here'''
    args = read_args()
    print("Inference on a patient MRI scan")
    print("--"* 50)
    device, model, _, _, _, _, _, post_trans= model_loss_optim(1, 1e-3, 1e-5)
    diagnose(model, args.weights, args.patient, device, post_trans)
    print("--"*50)

    print('Show labled image ..')
    # show_labeled_image(args.patient)
    print('done!!')

if __name__ == "__main__":
    main()


