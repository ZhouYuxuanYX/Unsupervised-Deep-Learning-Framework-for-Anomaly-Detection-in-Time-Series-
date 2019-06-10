from pathlib import Path
from model.Param import *
from Preprocess.Preprocesser import Preprocesser
from model.visualizing import plot_loss, plot_prediction
from model.Model import *
import os
import warnings
from anomaly_detection import anomaly_detection
from time_series_analysis import *
warnings.filterwarnings("ignore")

# Here the prediction[1] is to be used, cause it's real prediction
def mse_metric(train, predictions, prediction_steps):
    MSE = []
    for file in range(len(predictions[1])):
        # if the sliding_step is large, maybe the last few points on the end could not be covered(less than the sliding_step)
        error = train[file][:len(predictions[1][file])]-predictions[1][file]
        # Median value is more stable than mean
        mse = np.median((error)**2)
        MSE.append(mse)
    return MSE

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

# load the parameters for the experiment params.json file in model dir
model_dir = os.path.join(general_settings.experiments_path, general_settings.model_type)
json_path = Path(model_dir) / 'params.json'
params_train = Params.update(json_path)

##### handle the data #####
# preprocess the data
preprocesser = Preprocesser.from_tdm(general_settings.data_path, general_settings.channels, general_settings.time_index)
data_preprocessed_labeled = preprocesser.data

# select the channel out of the whole array
data_all_files = preprocesser.select_channel(channel_name)

# even if read just one file, indexing like e.g. 3:4 should be used in order to keep the outer []
train = data_all_files[:2]

# train and evaluate the model setting
if general_settings.model_type == "1d conv":
    models, loss, predictions = Convolutioanl_autoencoder.train_and_predict(params_train, train, general_settings)
elif general_settings.model_type == "MLP":
    models, loss, predictions = Multilayer_Perceptron.train_and_predict(params_train, train, general_settings)
elif general_settings.model_type == "wavenet":
    models, loss, predictions = Wavenet.train_and_predict(params_train, train, general_settings)
else:
    models, loss, predictions = Variational_Autoecnoder.train_and_predict(params_train, train, general_settings)

MSE = mse_metric(train, predictions, general_settings.prediction_steps)
anomaly_detection(train, predictions, general_settings.detection_mode)