
## Brain tumors segmentation
The advancement in healthcare and biotechnology have led to the growing use and need of AI in medical imaging analysis.
AI technique deep learning has led to massive development in the field of image recognition.
There is a lot of research and development going on in the field of medical imaging analysis using deep learning.
It is neccessary to develop an automnomous technique which could label disease in images, therfore, it can 
help physicians in diagnosing disease, for instance brain tumors. In this repository, I used SegResNet model to recognize 
brain tumors. The model has been traind on Brats-2020 dataset, which contains 3D brain images of different modalities such as T1, T2, T1ce and Flair. 
The dataset also contains labels for each MRI image.

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
To visualize modalities with labels, you can use following command
```
from utils import util
import configs

%matplotlib inline
util.inspect_data(configs.Config.a_test_patient)
```
you will see:
![plot]('brats-16-17 Results'/brats_data_inspection.png)

