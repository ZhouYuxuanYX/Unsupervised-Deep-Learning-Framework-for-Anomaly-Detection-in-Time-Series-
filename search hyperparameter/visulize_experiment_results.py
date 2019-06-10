from pathlib import Path
from model.Param import *
import os
import numpy as np
from model.visualizing import plot_loss, plot_prediction
from anomaly_detection import anomaly_detection

# define paths
# use the Python3 Pathlib modul to create platform independent path
general_settings = Params.update(
    Path("C:/Users/zhouyuxuan/PycharmProjects/Masterarbeit/experiments/general_settings.json"))

model_dir = os.path.join(general_settings.experiments_path, "MLP")
experiment_dir = os.path.join(model_dir, "num_epochs_pred_step_1")

for channel_name in general_settings.channels:
    # Check every subdirectory of parent_dir
    for subdir in os.listdir(experiment_dir):
        loss = np.load(os.path.join(experiment_dir,subdir,channel_name+"_loss.npy"))
        predictions = np.load(os.path.join(experiment_dir, subdir, channel_name+"_predictions.npy"))
        train = np.load(os.path.join(experiment_dir, subdir,channel_name+"_train.npy"))
        plot_loss(loss)
        plot_prediction(train, predictions, 5) # choose prediction steps
        anomaly_detection(train, predictions, general_settings.detection_mode)