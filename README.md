[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/faizan1234567/Brats-20-Tumors-segmentation/blob/main/notebooks/BraTS21_setup.ipynb)]
# Brain Tumors Segmentation on BraTS21 Dataset
Brain tumor is one of the deadlist kind of disease around the global. Among these, gliomas are the most common type. Radiologists use MRI images to diagnose the disease. Radiologists can sometimes make errors, and it's highly time consuming process. To assit radioligits, deep learning is being used. Therefore, to train a deep learning model for diagnosing the disease, a large dataset is being used to train the AI model for accurate predictions.
![Alt Text](https://github.com/faizan1234567/Brats-20-Tumors-segmentation/blob/main/media/gif.gif)

## Dataset Description
All [BraTS21](http://braintumorsegmentation.org/) mpMRI scans are available as [NIfTI](https://radiopaedia.org/articles/nifti-file-format) files and described as  T2 Fluid Attenuated Inversion Recovery(Flair), native(T1), T2-weighted(T2), and post-contrast T1-weighted (T1Gd). They were acquired with differnt clinical protocals and various scanners from different institutions.

Annotations consistsof  GD-enhancing tumor (ET — label 4), the peritumoral edematous/invaded tissue (ED — label 2), and the necrotic tumor core (NCR — label 1), more detail here.

The dataset is represented in a directory in the following sturcture, please make necessary changes if required in your case.
```
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
```

![alt text](https://github.com/faizan1234567/Brats-20-Tumors-segmentation/blob/main/media/fig_brats21.png)
image from  [Baid et al.](https://arxiv.org/pdf/2107.02314v1.pdf)

## Installation
```
git clone https://github.com/faizan1234567/Brats-20-Tumors-segmentation.git
cd Brats-20-Tumors-segmentation

python3 -m venv segmentation #linux
source segmentation/bin/activate
  
python3 -m venv segmentation #windows
 .\segmentation\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
To train on Brats21, run the training command
First configure some paths in configs.py, these are some necessary paths about the dataset directory, data loading, and other important parameters.
```
python train.py -h
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           dataset root directory path
  --fold FOLD           folder name
  --json_file JSON_FILE
                        path to json file
  --batch BATCH         batch size
  --img_roi IMG_ROI     image roi size
  --val_every VAL_EVERY
                        validate every 2 epochs
  --max_epochs MAX_EPOCHS
                        maximum number of epoch to train
  --workers WORKERS     Number of data loading workers
  --pretrained_model PRETRAINED_MODEL
                        path to pretraiend model
  --pretrained          use pretrained weights.
  --resume              starting training from the saved ckpt.
  --platform_changed    pc changed, so that dataset dir has been set accordingly

python train.py --data <dataset-dir> --fold 0 --json_file <dataset-split-file-path>
 --batch 1 --workers 8 --val_every 2 --max_epochs 100 --pretrained_model <pretrained-model-path> 
   ```
To test the model:
```
python test.py -h
optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS     weight file path
  --fold FOLD           fold number for evaluation
  --workers WORKERS     number of workers
  --batch BATCH         batch size to load the dataset
  --json_file JSON_FILE
                        path to the data json file
  --platform_changed    running on other platfrom

python test.py --weights <path-to-weights>  --fold 0 --workers 2 --batch 1 --json_file <dataset-split-file-path>
   ```
To visualize, use:
```
python show.py -h
optional arguments:
  -h, --help            show this help message and exit
  --json_file JSON_FILE
                        dataset split file
  --fold FOLD           folder number
  --phase PHASE         validation or training.
  --save SAVE           results save directory
  --get_abnormal_area   get full abnormal are
  --visualize_data_gif  visulize data gif, and create a gif file
  --visualize_data_sample
                        visualize one sample

python show.py --json_file <path> --fold 0 --phase <"val"> --save <path> 
```

## References
[1]. http://braintumorsegmentation.org/

[2]. https://monai.io/

[3]. https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21

[4]. https://github.com/amanchadha/coursera-ai-for-medicine-specialization

[5]. https://arxiv.org/pdf/1906.01796v2.pdf

[6]. https://arxiv.org/abs/2103.14030

[7]. https://arxiv.org/abs/1512.03385