import numpy as np
import os
import pandas as pd
from model.Param import *
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from model.visualizing import plot_loss, plot_prediction
from model.Model import *
from pathlib import Path
from matplotlib2tikz import save as tikz_save
from scipy.stats import norm

def reconstruction_error(inputs, outputs):
    """Return the mean square errors"""
    inputs = np.array(inputs)
    # inputs = pd.DataFrame(inputs)
    # inputs = inputs.rolling(2, win_type='triang').mean().fillna(method='backfill').values.squeeze()
    outputs = np.array(outputs)
    error_reconstructed = inputs - outputs
    return  error_reconstructed

def control_limits(variance_estimation, test_predicted):
    upper_control_limit = np.array(test_predicted) + 3 * (np.array(variance_estimation[0:-1]) ** (1 / 2)) # [0:-1] not using the error in current step
    lower_control_limit = np.array(test_predicted) - 3 * (np.array(variance_estimation[0:-1]) ** (1 / 2))
    return upper_control_limit, lower_control_limit

rs = RobustScaler()

def rob_scal(data):
    """Apply the RobustScaler to depress the effect of outliers

    Args:
        data: 2D array [examples, features],
        becasue it's raw data, so it should be seen as n examples with only one feature(each point)

    Returns:
        signal_scaled: List
    """
    signal = data.reshape(-1,1) # It returns the reshaped array but doesn't change the original one
    signal_scaled = rs.fit_transform(signal)
    signal_scaled = signal_scaled.reshape(1, -1).squeeze()
    return signal_scaled

# define paths
# use the Python3 Pathlib modul to create platform independent path
general_settings = Params.update(
    Path("C:/Users/zhouyuxuan/PycharmProjects/Masterarbeit/experiments/general_settings.json"))

# load the parameters for the experiment params.json file in model dir
model_dir = os.path.join(general_settings.experiments_path, general_settings.model_type)
json_path = Path(model_dir) / 'params.json'
params_train = Params.update(json_path)

ResultPath = "D:/NAB-master/NAB-master/results/1dConv"
DataPath = "D:/NAB-master/NAB-master/data"
for diret in os.listdir(DataPath):
    if os.path.isdir(os.path.join(DataPath,diret)):
        for file in os.listdir(os.path.join(DataPath,diret)):
            df = pd.read_csv(os.path.join(DataPath,diret,file))
            signal = df["value"].values
            input = rob_scal(signal)
            train = np.reshape(input,(1,-1))

            models, loss, predictions = Convolutioanl_autoencoder.train_and_predict(params_train, train,
                                                                                general_settings)

            # models, loss, predictions = Wavenet.train_and_predict(params_train, train, general_settings)

            error = predictions[1][0]
            prediction = predictions[1][0]
            data = train.squeeze()[:len(prediction)]
            error_data = reconstruction_error(data, prediction)
            mu, std = norm.fit(error_data)
            plt.figure()
            plt.plot(error_data)
            plt.figure()
            plt.plot(data)
            plt.plot(prediction)
            plt.legend(["preprocessed data", "prediction"])
            LCL = prediction + mu - 6 * std
            UCL = prediction + mu + 6 * std
            plt.figure()
            plt.plot(data)
            mask_anomaly = (LCL > data).astype(int) + (UCL < data).astype(int)
            anomaly = mask_anomaly * data
            x = np.array(range(len(anomaly)))[anomaly != 0]
            y = anomaly[anomaly != 0]
            plt.plot(x, y, 'rx')
            plt.legend(["original signal", "anomalies"], loc='upper center', bbox_to_anchor=(0.5, -0.15))
            plt.fill_between(list(range(len(prediction))), UCL, LCL, color='k', alpha=.25)
            score = pd.DataFrame(mask_anomaly, columns=["anomaly_score"])
            result = pd.DataFrame(mask_anomaly, columns=["label"])
            output = pd.concat([df,score,result],axis=1).fillna(0)
            DirPath = os.path.join(ResultPath,diret)
            if not os.path.exists(DirPath):
                os.makedirs(DirPath)
            output.to_csv(os.path.join(DirPath,"1dConv_"+file), index=False)
