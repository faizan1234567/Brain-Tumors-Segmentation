'''Inference on 3D brain tumor image

The model will take an image and the code will show the mask and true labels on the image
'''

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from segment_3d import inference
import argparse
from configs import Config
from BratsCustom import BratsDataset20
from segment_3d import model_loss_optim


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default= Config.a_test_patient, type = str, help = "path to the case directory in the test/val/train dir")
    parser.add_argument("weights", default= " ", type = str, help= "weight file path")
    opt = parser.parse_args()
    return opt

def load_image(patient_path):
    """load the image from the given path
    
    Args:
    patient_path: os.path (str)"""
    dataset = BratsDataset20(phase="val", patient=patient_path)
    img_data = dataset.__getitem__()
    scan = img_data["image"]
    mask = img_data["mask"]
    return (scan, mask)



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
   
    model.load_state_dict(torch.load(weights))
    # torch.load(os.path.join(weights, "best_metric_model.pth")))
    
    model.eval()
    with torch.no_grad():
    # select one image to evaluate and visualize the model output
        # val_input = patient[6]["image"].unsqueeze(0).to(device)
        print("loading patient file...")
        val_input, mask = load_image(patient)
        print("Success: the patient file loaded.")
        print("Shape of the scan: {}, and shape of mask: {}".format(val_input.shape, mask.shape))
        roi_size = (128, 128, 64)
        sw_batch_size = 1
        val_output = inference(val_input, model)
        val_output = post_trans(val_output[0])
        plt.figure("image", (24, 6))
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.title(f"image channel {i}")
            plt.imshow(val_input[i, :, :, 70].detach().cpu(), cmap="gray")
        plt.savefig('image_channels.png')
        plt.show()
        # visualize the 3 channels label corresponding to this image
        plt.figure("label", (24, 6))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(f"label channel {i}")
            plt.imshow(mask[i, :, :, 70].detach().cpu())
        plt.savefig('labels_channel.png')
        plt.show()
        # visualize the 3 channels model output corresponding to this image
        plt.figure("output", (24, 6))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(f"output channel {i}")
            plt.imshow(val_output[i, :, :, 70].detach().cpu())
        plt.savefig('output_channels.png')
        plt.show()

def main():
    '''all the code goes here'''
    args = read_args()
    print("Inference on a patient MRI scan")
    print("--"* 50)
    device, model, _, _, _, _, _, post_trans= model_loss_optim(1, 1e-3, 1e-5)
    diagnose(model, args.weights, args.patient, device, post_trans)
    print("--"*50)
    print('done!!')

if __name__ == "__main__":
    main()


