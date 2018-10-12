import matplotlib.pyplot as plt

def plot_loss(loss, training_mode):
    if training_mode == "offline":
        for file in range(len(loss[0])):
            plt.figure()
            plt.plot(loss[0][file])
            plt.plot(loss[1][file])
            plt.legend(["train loss", "validation loss"])
            plt.title("file {}".format(file))
    else:
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

def plot_prediction(train, val, predictions, training_mode):
    for file in range(len(predictions[0])):
        if training_mode == "online":
            prediction_train = [0]
            plt.figure()
            prediction_train.extend(predictions[0][file])
            plt.plot(prediction_train)
            # because predictions are one step ahead, at time zero, there is no history thus no prediction
            plt.plot(train[file])
            prediction = [0]
            prediction.extend(predictions[1][file])
            plt.plot(prediction)
            plt.legend(["prediction on train(lagged) data","data","prediction"])
            # plt.ion()
            # plt.show()
            # plt.pause(0.001)

        else:
            prediction_train = [0]
            plt.figure()
            prediction_train.extend(list(predictions[0][file]))
            plt.plot(prediction_train)
            plt.plot(train[file])
            plt.legend(["prediction on train set", "train data"])

            plt.figure()
            prediction_val = [0]
            prediction_val.extend(list(predictions[1][file]))
            plt.plot(prediction_val)
            # every time, the validation file is the same
            plt.plot(val)
            plt.legend(["prediction on validation set", "validation data"])