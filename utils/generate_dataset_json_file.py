# import necessary modules..
import json
import sys
import os
from pathlib import Path
import random
import logging

#setting up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("json_file_creation.log")
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)
# add to path
try:
    from config import configs
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1] #project root directory
    if ROOT not in sys.path:
        sys.path.append(str(ROOT))
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path
    from config import configs

# store the ttraining dataset into json file
def generate_json_file(path, name):
    """
    Generate a json file to hold the dataset records

    Parameters
    ----------
    path: str
    name: str
    """
    logger.info("Configuring training path and shuffling patients record")
    json_file_path = os.path.join(path, name)
    is_file = os.path.isfile(json_file_path)
    random.seed(50) # to generate consistent random results
    train_root_dir = configs.Config.newGlobalConfigs.train_root_dir
    patients_records = os.listdir(train_root_dir)
    random.shuffle(patients_records)

    dataset = dict()
    dataset['training'] = []
    fold = 0
    logger.info("Running on patients folder and creating a json file to split data into 5 folds")
    for i, patient_record in enumerate(patients_records):
        if i % 250 == 0 and i != 0:
            fold +=1 
        patient = dict()
        patient["fold"] = fold
        patient_path = train_root_dir + "/" + patient_record
        patient_files = os.listdir(patient_path)
        patient["image"] = []
        for img in patient_files:
            if not "-seg.nii.gz" in img:
                patient["image"].append(patient_path + "/" + img)
            else:
                patient["label"] = patient_path + "/" + img
        dataset['training'].append(patient)
        logger.info("A patient record added: {}".format(patient))           
    logger.info('Write everything into a json file')
    if (not is_file):
        with open(json_file_path, 'w') as file:
            json.dump(dataset, file)
    else:
        logger.info('The json file already exists')
        

if __name__ == "__main__":
    logger.info("Running Everything now")
    generate_json_file(configs.Config.newGlobalConfigs.root_dir, 'dataset.json')
    logger.info('done')




