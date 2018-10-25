import warnings
from utils import save_dict_to_json
from Preprocesser import Preprocesser
from visualize import plot_loss, plot_prediction
from Model import Convolutioanl_autoencoder, Multilayer_Perceptron, Variational_Autoecnoder, Wavenet
import os
import numpy as np
warnings.filterwarnings("ignore")

def mse_metric(train, predictions, prediction_steps):
    MSE = []
    for file in range(len(predictions[0])):
        # if the sliding_step is large, maybe the last few points on the end could not be covered(less than the sliding_step)
        error = train[file][prediction_steps:prediction_steps+len(predictions[1][file])]-predictions[1][file]
        # Median value is more stable than mean
        mse = np.median((error)**2)
        MSE.append(mse)
    return MSE

def train_and_evaluate(params_train, general_settings, job_dir):

    ##### handle the data #####
    # preprocess the data
    preprocesser = Preprocesser.from_tdm(general_settings.data_path, general_settings.channels, general_settings.time_index)

    # save the preprocessed data in a numpy array
    preprocesser.save(os.path.join(general_settings.processed_data_path, 'preprocessed data array'))
    for channel_name in general_settings.channels:
        # select the channel out of the whole array
        data_all_files = preprocesser.select_channel(channel_name)

        # even if read just one file, indexing like e.g. 3:4 should be used in order to keep the outer []
        train = data_all_files[0:5]

        # train and evaluate the model setting
        if general_settings.model_type == "1d conv":
            models, loss, predictions = Convolutioanl_autoencoder.train_and_predict(params_train, train, general_settings)
        elif general_settings.model_type == "MLP":
            models, loss, predictions = Multilayer_Perceptron.train_and_predict(params_train, train, general_settings)
        elif general_settings.model_type == "wavenet":
            models, loss, predictions = Wavenet.train_and_predict(params_train, train, general_settings)
        else:
            models, loss, predictions = Variational_Autoecnoder.train_and_predict(params_train, train, general_settings)

        models[0].save(os.path.join(job_dir,channel_name+"_Model.h5"))
        np.save(os.path.join(job_dir,channel_name+"_loss"), loss)
        np.save(os.path.join(job_dir,channel_name+"_predictions"), predictions)
        np.save(os.path.join(job_dir, channel_name+"_train"), train)
        plot_loss(loss)
        plot_prediction(train,predictions, general_settings.prediction_steps)
        MSE = mse_metric(train, predictions, general_settings.prediction_steps)
        metric = {channel_name+"_file"+str(i):mse for i, mse in enumerate(MSE)}
        json_path = os.path.join(job_dir, channel_name+"_metric.json")
        save_dict_to_json(metric, json_path)












