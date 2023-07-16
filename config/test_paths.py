import os

import configs
# path to test
path = configs.Config.newGlobalConfigs.train_root_dir

## ummcoment for path test
# check the path exits or not, if not correct it.
# if os.path.exists(path):
#     print("yes: the path found")
# else:
#     print("No: the path doesnt exists")
files = os.listdir(path)