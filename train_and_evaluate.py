import warnings
from utils import save_dict_to_json
from Preprocesser import Preprocesser
from visualize import plot_loss, plot_prediction
from Model import Convolutioanl_autoencoder, Multilayer_Perceptron, Variational_Autoecnoder
import os
warnings.filterwarnings("ignore")

def mse_metric(train, predictions):
    MSE = 0
    for file in range(len(predictions[0])):
        mse = ((train[file][1:]-predictions[1][file])**2).sum()
        MSE += mse
    return MSE

def train_and_evaluate(params_train, channel_name, general_settings, job_dir):

    ##### handle the data #####
    # preprocess the data
    preprocesser = Preprocesser.from_tdm(general_settings.data_path, general_settings.channels, general_settings.time_index)
    data_preprocessed_labeled = preprocesser.data

    # save the preprocessed data in a numpy array
    preprocesser.save(os.path.join(general_settings.processed_data_path, 'preprocessed data array'))

    # select the channel out of the whole array
    data_all_files = preprocesser.select_channel(channel_name)

    # even if read just one file, indexing like e.g. 3:4 should be used in order to keep the outer []
    train = data_all_files[0:5]

    # choose a verified file to be the validation set for all the model
    validation = None

    # train and evaluate the model setting
    models, loss, predictions = Convolutioanl_autoencoder.train_and_predict(params_train, train, validation)
    plot_loss(loss, params_train.training_mode)
    plot_prediction(train, validation, predictions, params_train.training_mode)
    metric = {"mse":mse_metric(train, predictions)}
    json_path = os.path.join(job_dir, "metric.json")
    save_dict_to_json(metric, json_path)












