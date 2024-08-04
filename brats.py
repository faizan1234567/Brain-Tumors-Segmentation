"""
==========================
Data loading and processing
===========================

credit: https://github.com/faizan1234567/CKD-TransBTS/blob/main/BraTS.py
"""
import torch
import os
from torch.utils.data.dataset import Dataset
from utils.all_utils import pad_or_crop_image, minmax, load_nii, pad_image_and_label, listdir, get_brats_folder

class BraTS(Dataset):
    def __init__(self, patients_dir, patient_ids, mode, target_size = (128, 128, 128)):
        super(BraTS,self).__init__()
        self.patients_dir = patients_dir
        self.patients_ids = patient_ids
        self.mode = mode
        self.target_size = target_size
        self.datas = []
        self.pattens =["-t1n","-t1c","-t2w","-t2f"]
        if mode == "train" or mode == "train_val" or mode == "test" :
            self.pattens += ["-seg"]
        for patient_id in patient_ids:
            paths = [f"{patient_id}{patten}.nii.gz" for patten in self.pattens]
            patient = dict(
                id=patient_id, t1=paths[0], t1ce=paths[1],
                t2=paths[2], flair=paths[3], seg=paths[4] if mode == "train" or mode == "train_val" or mode == "val" or mode == "test" or mode == "visualize" else None
            )
            self.datas.append(patient)

    def __getitem__(self, idx):
        patient = self.datas[idx]
        patient_id = patient["id"]
        crop_list = []
        pad_list = []
        patient_image = {key:torch.tensor(load_nii(f"{self.patients_dir}/{patient_id}/{patient[key]}")) for key in patient if key not in ["id", "seg"]}
        patient_label = torch.tensor(load_nii(f"{self.patients_dir}/{patient_id}/{patient['seg']}").astype("int8"))
        patient_image = torch.stack([patient_image[key] for key in patient_image])  
        if self.mode == "train" or self.mode == "train_val" or self.mode == "test":
            # ET=3, ED=2, NCR=1, and background=0
            et = patient_label == 3
            tc = torch.logical_or(patient_label == 1, patient_label == 3)
            wt = torch.logical_or(tc, patient_label == 2)
            patient_label = torch.stack([et, tc, wt])

        # Removing black area from the edge of the MRI
        nonzero_index = torch.nonzero(torch.sum(patient_image, axis=0)!=0)
        z_indexes, y_indexes, x_indexes = nonzero_index[:,0], nonzero_index[:,1], nonzero_index[:,2]
        zmin, ymin, xmin = [max(0, int(torch.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
        zmax, ymax, xmax = [int(torch.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
        patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax].float()

        for i in range(patient_image.shape[0]):
            patient_image[i] = minmax(patient_image[i])
        
        patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
        if self.mode == "train" or self.mode == "train_val" :
            patient_image, patient_label, pad_list, crop_list = pad_or_crop_image(patient_image, patient_label, target_size=self.target_size)
        elif self.mode == "test":
            d, h, w = patient_image.shape[1:]
            pad_d = (128-d) if 128-d > 0 else 0
            pad_h = (128-h) if 128-h > 0 else 0
            pad_w = (128-w) if 128-w > 0 else 0
            patient_image, patient_label, pad_list = pad_image_and_label(patient_image, patient_label, target_size=(d+pad_d, h+pad_h, w+pad_w))

        return dict(
            patient_id = patient["id"],
            image = patient_image,
            label = patient_label,
            nonzero_indexes = ((zmin, zmax), (ymin, ymax), (xmin, xmax)),
            box_slice = crop_list,
            pad_list = pad_list
        )

    def __len__(self):
        return len(self.datas)

def get_datasets(dataset_folder, mode, target_size = (128, 128, 128)):
    dataset_folder = get_brats_folder(dataset_folder, mode)
    assert os.path.exists(dataset_folder), "Dataset Folder Does Not Exist1"
    patients_ids = [x for x in listdir(dataset_folder)]
    return BraTS(dataset_folder, patients_ids, mode, target_size=target_size)