import matplotlib.pyplot as plt
import numpy as np

def plot_loss(loss):
        for file in range(len(loss[0])):
            plt.figure()
            plt.title("file {}".format(file))
            plt.subplot(2,1,1)
            plt.plot(loss[0][file])
            plt.legend("train loss")

            plt.subplot(2,1,2)
            plt.plot(loss[1][file])
            plt.legend("rolling loss")

            # plt.ion()
            # plt.show()
            # plt.pause(0.001)

def plot_prediction(train, predictions,prediction_steps):
    for file in range(len(predictions[0])):
        plt.figure()
        plt.plot( predictions[0][file])
        plt.plot(train[file])
        plt.plot(predictions[1][file])
        # prediction = [0]*prediction_steps
        # prediction.extend(predictions[1][file])
        # plt.plot(prediction)
        plt.legend(["prediction on train(lagged) data","data","prediction"])
        # plt.ion()
        # plt.show()
        # plt.pause(0.001)

