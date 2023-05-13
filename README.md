
# Brain Tumors Segmentation on BraTS21 Dataset
Brain tumor is one of the deadlist kind of disease around the global. Among these, gliomas are the most common type. Radiologists use MRI images to diagnose the disease. Radiologists can sometimes make errors, and it's highly time consuming process. To assit radioligits, deep learning is being used. Therefore, to train a deep learning model for diagnosing the disease, a large dataset is being used to train the AI model for accurate predictions.
![Alt Text](https://github.com/faizan1234567/Brats-20-Tumors-segmentation/blob/main/media/gif.gif)

## Dataset Description
All [BraTS21](http://braintumorsegmentation.org/) mpMRI scans are available as [NIfTI](https://radiopaedia.org/articles/nifti-file-format) files and described as  T2 Fluid Attenuated Inversion Recovery(Flair), native(T1), T2-weighted(T2), and post-contrast T1-weighted (T1Gd). They were acquired with differnt clinical protocals and various scanners from different institutions.

Annotations consistsof  GD-enhancing tumor (ET — label 4), the peritumoral edematous/invaded tissue (ED — label 2), and the necrotic tumor core (NCR — label 1), more detail here.

The dataset is represented in a directory in the following sturcture, please make necessary changes if required in your case.

BraTS21/
├── train/
│   ├── RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/
│   │   ├── BraTS2021_00000/
│   │   │   ├── BraTS2021_00000_flair.nii.gz
│   │   │   ├── BraTS2021_00000_seg.nii.gz
│   │   │   ├── BraTS2021_00000_t1.nii.gz
│   │   │   ├── BraTS2021_00000_t2.nii.gz
│   │   │   └── BraTS2021_00000_t1ce.nii.gz
│   │   └── ...
│   ├── brats21_folds.json
│   └── BraTS21-17_Mapping.csv
└── validation/
    └── RSNA_ASNR_MICCAI_BraTS2021_ValidationData/
        ├── BraTS2021_00001/
        │   ├── BraTS2021_00001_flair.nii.gz
        │   ├── BraTS2021_00001_t1.nii.gz
        │   ├── BraTS2021_00001_t1ce.nii.gz
        │   └── BraTS2021_00001_t2.nii.gz
        └── ...

![alt text](https://github.com/faizan1234567/Brats-20-Tumors-segmentation/blob/main/media/fig_brats21.png)
image from  [Baid et al.](https://arxiv.org/pdf/2107.02314v1.pdf)

## Installation
```
git clone https://github.com/faizan1234567/Brats-20-Tumors-segmentation.git
cd Brats-20-Tumors-segmentation
   ```
Create a virtual environment on linux or Windows to install neccessary packages
```
python3 -m venv segmentation
source segmentation/bin/activate
   ```
And for Windows.
```
python3 -m venv segmentation
 .\segmentation\Scripts\activate

   ```
Now install the required packages by running the following command
```
pip install --upgrade pip
pip install -r requirements.txt
   ```
and check everything has installed?
```
pip list
```
You will see all the packages have been installed.



## Usage
To train on Brats2020, run the training command
First configure some paths in configs.py, these are some necessary paths about the dataset directory 
and data loading.

```
python segment_3d.py -h
python segment_3d.py --epochs 200 --lr 0.0001 --weight_decay 1e-5
 --batch 1 --workers 8 --data_dir path_to_configure --name dir_name_to_configure
   ```
To test the model:
```
python test.py -h
python test.py --weights path_to_configure --workers 2 --batch 1
   ```

To visualize modalities with labels, you can use following command
```
from utils import util
import configs

%matplotlib inline
util.inspect_data(configs.Config.a_test_patient)
```
you will see:
![brats_data_inspection](https://user-images.githubusercontent.com/61932757/189496106-754d73c9-d90d-4aae-a008-ca0384b2cbfc.png)


