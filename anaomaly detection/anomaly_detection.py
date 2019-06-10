import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from detection_modes import arima
from matplotlib2tikz import save as tikz_save

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

def anomaly_detection(train, predictions, detection_mode):
    for counter, file in enumerate(range(len(predictions[0]))):
        prediction = predictions[1][file]
        data = train[file][:len(prediction)]
        error_data = reconstruction_error(data, prediction)
        plt.figure()
        plt.plot(error_data)
        # ts_analysis(error_data)
        if detection_mode == "ARIMA":

            arima(error_data, data, prediction)


        if detection_mode == "Gaussian":
            # calculate the threshold by fitting a gaussian model
            # fit a normal distribution
            mu, std = norm.fit(error_data)
            # plt.figure()
            # # plot the histogram
            # plt.hist(error_data, bins = 25, density=True, alpha=0.6, color='g')
            # # plot the pdf
            # # Plot the PDF.
            # xmin, xmax = plt.xlim()
            # x = np.linspace(xmin, xmax, 100)
            # p = norm.pdf(x, mu, std)
            # plt.plot(x, p, 'k', linewidth=2)
            # title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
            # plt.title(title)
            # draw confidence bound(gray)
            LCL = prediction + mu-3*std
            UCL = prediction + mu+3*std
            plt.figure()
            plt.plot(data)
            mask_anomaly = (LCL > data).astype(int) + (UCL < data).astype(int)
            anomaly = mask_anomaly * data
            x = np.array(range(len(anomaly)))[anomaly != 0]
            y = anomaly[anomaly != 0]
            plt.plot(x, y, 'rx')
            plt.legend(["original signal","anomalies"],loc='upper center', bbox_to_anchor=(0.5,-0.15))
            plt.fill_between(list(range(len(prediction))), UCL, LCL, color='k', alpha=.25)
            # style in plot can not be displayed by matplotlib2tikz!!!!
            # If we plot the boundary, sometimes it will be too large to show in the plot(on anomalous points)
            tikz_save("wavefile{}.tex".format(counter))




