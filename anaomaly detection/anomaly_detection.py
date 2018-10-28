import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from detection_modes import arima
from time_series_analysis import ts_analysis

def reconstruction_error(inputs, outputs):
    """Return the mean square errors"""
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    error_reconstructed = inputs - outputs
    return  error_reconstructed

def control_limits(variance_estimation, test_predicted):
    upper_control_limit = np.array(test_predicted) + 3 * (np.array(variance_estimation[0:-1]) ** (1 / 2)) # [0:-1] not using the error in current step
    lower_control_limit = np.array(test_predicted) - 3 * (np.array(variance_estimation[0:-1]) ** (1 / 2))
    return upper_control_limit, lower_control_limit

# def plot_anomaly(data, prediction):
#     plt.figure()
#     plt.plot(data)
#     plt.plot(prediction)
#     plt.legend(["data", "prediction"])
#
#     error_reconstructed = reconstruction_error(data, prediction)
#     plt.fill_between(list(range(len(prediction))), UCL, LCL, color='k', alpha=.25)

    # else:
    # plt.figure()
    # plt.plot(predictions[0][file])
    # plt.plot(train[file])
    # plt.legend(["prediction on train set", "train data"])
    #
    # plt.figure()
    # plt.plot(predictions[1][file])
    # # every time, the validation file is the same
    # plt.plot(val)
    # plt.legend(["prediction on validation set", "validation data"])


def anomaly_detection(train, predictions, detection_mode):
    for file in range(len(predictions[0])):
        if detection_mode == "ARIMA":
            data = train[file][1:]
            prediction = predictions[1][file]
            error_data = reconstruction_error(data, prediction)
            arima(error_data, data, prediction)

        # if detection_mode == "Gaussian":
        #     data = train[file][1:]
        #     prediction = predictions[1][file]
        #     error_data = reconstruction_error(data, prediction)
        #     mse = error_data**2
        #     # calculate the threshold by fitting a gaussian model
        #     # fit a normal distribution
        #     mu, std = norm.fit(mse)
        #     # plot the histogram
        #     plt.hist(mse, bins = 25, density=True, alpha=0.6, color='g')
        #     # plot the pdf
        #     # Plot the PDF.
        #     xmin, xmax = plt.xlim()
        #     x = np.linspace(xmin, xmax, 100)
        #     p = norm.pdf(x, mu, std)
        #     plt.plot(x, p, 'k', linewidth=2)
        #     title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
        #     plt.title(title)
        #     # draw confidence bound(gray)
        #     LCL = prediction - np.sqrt(mu+2*std)
        #     UCL = prediction + np.sqrt(mu+2*std)
        #     plt.figure()
        #     plt.fill_between(prediction, LCL, UCL, color='g')
        #     # create a mask array for anomaly points
        #     mask_anomaly = (LCL>data).astype(int)+(UCL<data).astype(int)
        #     # anomaly =

