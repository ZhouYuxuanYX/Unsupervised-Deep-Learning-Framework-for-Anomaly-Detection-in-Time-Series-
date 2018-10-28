import pandas as pd
import numpy as np

def to_three_d_array(lists):
    """Expand a list to 3D array

    Args: List [Example1, Example2......](outer [] as first dimensions when transformed to array

    Returs: 3D array [examples, features, channels]
    """
    arrays = np.array(lists)
    arrays = np.reshape(arrays,(arrays.shape[0],-1,1))
    return arrays

def create_lagged_df(data, lags):
    data = pd.DataFrame(data)
    df = pd.concat([data.shift(lag) for lag in range(-lags,0)], axis=1)
    df.columns = ['lag {}'.format(-lag) for lag in range(-lags,0)]
    data_combined = data.join(df)
    # Padded in the left, in order to synchronize with the original data
    data_combined = data_combined.fillna(0).values
    return data_combined

def create_callbacks(callbacks):
    if callbacks == None:
        CallBacks = None
    else:
        CallBacks = []
        for CallBack in callbacks:
            if CallBack == "early stopping":
                from keras.callbacks import EarlyStopping
                # Because stochastic gradient descent is noisy, patience must be set to a relative large number
                early_stopping_monitor = EarlyStopping(patience=10)
                CallBacks.append(early_stopping_monitor)

            if CallBack == "TensorBoard":
                from keras.callbacks import TensorBoard
                log_dir = 'C:/Users/zhouyuxuan/PycharmProjects/Masterarbeit/experiments/logdir'
                CallBacks.append(TensorBoard(log_dir=log_dir))
    return CallBacks
