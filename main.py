from pathlib import Path
from utils import *

##### Initializing #####

# define paths
# use the Python3 Pathlib modul to create platform independent path
general_settings = Params.update(
    Path("C:/Users/zhouyuxuan/PycharmProjects/Masterarbeit/experiments/general_settings.json"))

# choose the channel for training
channel_name = 'p_0'

# remark: for prediction-based model, they only use the lagged data as feature and the original data as label,
# for reconstruction-based, they also include the original data as feature as well as label
# and the number of features should be even number(multiple of cpus), if it's odd number, then Keras will raise error message

# load the parameters for the experiment params.json file in model path
json_path = Path(general_settings.model_path) / 'params.json'
params_train = Params.update(json_path)