[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Languages](https://img.shields.io/github/languages/top/faizan1234567/BraTS23-Tumors-Segmentation)](https://github.com/faizan1234567/BraTS23-Tumors-Segmentation)
[![License](https://img.shields.io/github/license/faizan1234567/BraTS23-Tumors-Segmentation)](https://github.com/faizan1234567/BraTS23-Tumors-Segmentation/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/faizan1234567/BraTS23-Tumors-Segmentation/build.yml?branch=main)](https://github.com/faizan1234567/BraTS23-Tumors-Segmentation/actions)
[![Issues](https://img.shields.io/github/issues/faizan1234567/BraTS23-Tumors-Segmentation)](https://github.com/faizan1234567/BraTS23-Tumors-Segmentation/issues)
[![Contributors](https://img.shields.io/github/contributors/faizan1234567/BraTS23-Tumors-Segmentation)](https://github.com/faizan1234567/BraTS23-Tumors-Segmentation/graphs/contributors)
[![Stars](https://img.shields.io/github/stars/faizan1234567/BraTS23-Tumors-Segmentation)](https://github.com/faizan1234567/BraTS23-Tumors-Segmentation/stargazers)
[![Forks](https://img.shields.io/github/forks/faizan1234567/BraTS23-Tumors-Segmentation)](https://github.com/faizan1234567/BraTS23-Tumors-Segmentation/network/members)
[![Last Commit](https://img.shields.io/github/last-commit/faizan1234567/BraTS23-Tumors-Segmentation)](https://github.com/faizan1234567/BraTS23-Tumors-Segmentation/commits/main)
[![Repo Size](https://img.shields.io/github/repo-size/faizan1234567/BraTS23-Tumors-Segmentation)](https://github.com/faizan1234567/BraTS23-Tumors-Segmentation)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/faizan1234567/Brats-20-Tumors-segmentation/blob/main/notebooks/BraTS21_setup.ipynb)
[![Open in Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-blue.svg)](https://www.kaggle.com/kernels/welcome)
[![Open in Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/faizan1234567/Brats-20-Tumors-segmentation/blob/main/notebooks/BraTS21_setup.ipynb)
## Brain Tumors Segmentation on BraTS23 Dataset (New)
Brain tumors are among the deadliest diseases worldwide, with gliomas being particularly prevalent and challenging to diagnose. Traditionally, physicians and radiologists rely on MRI and CT scans to identify and assess these tumors. However, this diagnostic process is not only time-consuming but also susceptible to human error, which can delay crucial treatment decisions.

To enhance diagnostic accuracy and efficiency, deep learning techniques are increasingly being integrated into medical imaging. Over the years, deep learning has demonstrated exceptional performance in analyzing complex medical images, providing reliable support to healthcare professionals. By leveraging large datasets, deep learning models can be trained to recognize patterns and anomalies in brain scans with reasonable accuracy, facilitating early detection and treatment of brain tumors.

This repository utilizes the ```BraTS 2021``` and ```BraTS 2023``` datasets to develop and evaluate both new and existing state-of-the-art algorithms for brain tumor segmentation. To facilitate research, we have made the code for training, evaluation, data loading, preprocessing, and model development open source. Researchers can use this template to build their models, enhancing accuracy and explainability.

![Alt Text](https://github.com/faizan1234567/Brats-20-Tumors-segmentation/blob/main/media/gif.gif)

### The Dataset 
All [BraTS23](http://braintumorsegmentation.org/) mpMRI scans are available as [NIfTI](https://radiopaedia.org/articles/nifti-file-format) files and include ```T2 Fluid Attenuated Inversion Recovery (FLAIR)```, ```native (T1)```, ```T2-weighted (T2)```, and post-contrast ```T1-weighted (T1Gd)``` images. These scans were acquired using different clinical protocols and various scanners from multiple institutions.

Annotations consist of GD-enhancing tumor (```ET — label 3```), peritumoral edematous/invaded tissue (```ED — label 2```), and necrotic tumor core (```NCR — label 1```). More details are available [here](https://www.synapse.org/#!Synapse:syn51156910/wiki/622351). These subregions can be clustered into three more segmentation-friendly regions which are used to evaluate the segmentation performance, including enhanced tumor (```ET```), tumor core (```TC```) (joining ```ET``` and ```NCR```), and whole tumor (```WT```) (joining ```ED``` to ```TC```).

The dataset contains ```1,251``` patient cases labeled by expert radiologists. However, cases in the validation and test sets are not annotated. Therefore, the actual training set is divided to training, validation, and test sets. The training set contains ```833```, validation and test sets contains ```209``` patient records each for model evaluation. [1]

```
└── dataset
    └── brats2023
        ├── train
        │   ├── BraTS-GLI-00000-000
        │   │   ├── BraTS-GLI-00000-000-seg.nii.gz
        │   │   ├── BraTS-GLI-00000-000-t1c.nii.gz
        │   │   ├── BraTS-GLI-00000-000-t1n.nii.gz
        │   │   ├── BraTS-GLI-00000-000-t2f.nii.gz
        │   │   └── BraTS-GLI-00000-000-t2w.nii.gz
        │   └── ...
        ├── val
        │   ├── BraTS-GLI-00006-000
        │   │   ├── BraTS-GLI-00006-000-seg.nii.gz
        │   │   ├── BraTS-GLI-00006-000-t1c.nii.gz
        │   │   ├── BraTS-GLI-00006-000-t1n.nii.gz
        │   │   ├── BraTS-GLI-00006-000-t2f.nii.gz
        │   │   └── BraTS-GLI-00006-000-t2w.nii.gz
        │   └── ...
        └── test
            ├── BraTS-GLI-00009-000
            │   ├── BraTS-GLI-00009-000-seg.nii.gz
            │   ├── BraTS-GLI-00009-000-t1c.nii.gz
            │   ├── BraTS-GLI-00009-000-t1n.nii.gz
            │   ├── BraTS-GLI-00009-000-t2f.nii.gz
            │   └── BraTS-GLI-00009-000-t2w.nii.gz
            └── ...
```

![alt text](https://github.com/faizan1234567/Brats-20-Tumors-segmentation/blob/main/media/fig_brats21.png)
image from  [Baid et al.](https://arxiv.org/pdf/2107.02314v1.pdf)

### Installation

```
git clone https://github.com/faizan1234567/BraTS23-Tumors-Segmentation
cd BraTS23-Tumors-Segmentation
```

create a virtual environment in Anaconda and activate it.
```
conda create -n brats_segmentation python=3.9.0 -y 
conda activate brats_segmentation
```
Now install all the dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

### Google Colab Usage
Upload the dataset on your drive. Update the dataset path in the configs according to the dataset split information shown above:
```python
from google.colab import drive
drive.mount('/gdrive')
```
clone the repo in colab notebook, and then install all the dependencies:
```
!pip install -r requirments.txt
```

### Usage
To train on Brats23, run the training command
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

### TODO
- [] MLOps tools integration such as Weight & Baises
- [] Multi-GPU training 
- [x] New Data augmentation options (private)
- [x] CKD-TransBTS and other SOTA semantic segmentation models to be added

### Cite

If you find this project useful in your research, please consider cite:

```latex
@misc{brats23-tumor-segmentation,
    title={Multi-modal BraTS 2023 brain tumor segmentation},
    author={Muhammad Faizan},
    howpublished = {\url{https://github.com/faizan1234567/Brats-20-Tumors-segmentation}},
    year={2023}
}
```

### Acknowledgements
[1]. http://braintumorsegmentation.org/

[2]. https://monai.io/

[3]. https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21

[4]. https://github.com/amanchadha/coursera-ai-for-medicine-specialization

[5]. https://arxiv.org/pdf/1906.01796v2.pdf

[6]. https://arxiv.org/abs/2103.14030

[7]. https://arxiv.org/abs/1512.03385
