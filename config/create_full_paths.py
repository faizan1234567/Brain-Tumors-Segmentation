
import yaml
import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(fmt= ("%(asctime)s: %(levelname)s: %(message)s" ), datefmt= "%d:%b:%y %H:%M:%S")
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

if __name__ == "__main__":
    # get command line args such as root dir and paths to other important files
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type =str, help = "path to root dir")
    parser.add_argument('--write', action="store_true", help="write the root dir to config.ymal?")
    args = parser.parse_args()

    # read ymal file
    with open("config/configs.yaml", "r") as f:
        configs = yaml.safe_load(f)
    
    logger.info(" Configs: {}".format(configs))

    # join some paths and update ymal file if they not exists already
    if args.configs:
        configs["config"]['dataset']['root_dir'] = args.root_dir
    else:
        pass
    
    # set full paths if needed
    try:
        full_paths = configs["full_paths"]
    except KeyError:
        root_dir = configs["config"]["dataset"]["root_dir"]
        train_dir = configs["config"]["dataset"]["train_sub_dir"]
        valid_dir = configs["config"]["dataset"]["validation_sub_dir"]
        dataset_excel_file = configs["config"]["dataset"]["dataset_xlsx"]
        a_test_patient = configs["config"]["dataset"]["a_test_patient"]
        json_file = configs["config"]["dataset"]["json_file"]

        # join paths
        train_path = root_dir + train_dir
        validation_path = root_dir + valid_dir
        dataset_file = root_dir + dataset_excel_file
        test_patient = train_dir + a_test_patient
        json_file_path = root_dir + json_file

        # write them in yaml file under full_paths heading
        configs['config']['full_paths'] = {}
        configs['config']['full_paths']['train_path'] = train_path
        configs['config']['full_paths']['validation_path'] = validation_path
        configs['config']['full_paths']['dataset_file'] = dataset_file
        configs['config']['full_paths']['test_patient'] = test_patient
        configs['config']['full_paths']['json_file'] = json_file_path

        # udpate config file
        with open('config/configs.yaml', 'w') as file:
            yaml.dump(configs, file)

    logger.info('full paths: {}'.format(configs["config"]["full_paths"]))
