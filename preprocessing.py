from sklearn.preprocessing import RobustScaler
import numpy as np

rs = RobustScaler()

def cut_head_tail(data, cut_num=50):
    """remove first/last few points

    Args:
        data: List with each channel as sublist. Each channel itself contains the signal in one file as 1D array

    Returns:
        data_cut: Same format as data, with head and tailed cut out
    """
    if cut_num == 0:
        data_cut = data
    else:
        data_cut = data[cut_num:-cut_num]
    return data_cut

def rob_scal(data):
    """Apply the RobustScaler to depress the effect of outliers

    Args:
        data: 2D array [examples, features],
        becasue it's raw data, so it should be seen as n examples with only one feature(each point

    Returns:
        signal_scaled: List
    """
    signal = data.reshape(-1,1) # It returns the reshaped array but doesn't change the original one
    signal_scaled = rs.fit_transform(signal)
    signal_scaled = signal_scaled.reshape(1, -1).squeeze()
    return signal_scaled

def to_three_d_array(lists):
    """Expand a list to 3D array

    Args: List [Example1, Example2......](outer [] as first dimensions when transformed to array

    Returs: 3D array [examples, features, channels]
    """
    arrays = np.array(lists)
    arrays = np.reshape(arrays,(arrays.shape[0],-1,1))
    return arrays

def window(scaled_data, win_size, step_size, overlap = True):
    """Apply sliding window to the data

    Args:
        scaled_data: 1D array [Example1, Example2......]

    Returns:
        1D array [Example1, Example2......]
    """
    windowed_data = []
    if overlap:
         for i in range(0, len(scaled_data), step_size):

            start = i + 1 - win_size if i + 1 - win_size >= 0 else 0
            observ = scaled_data[start:i+1]
            # Zero Padding left
            pad = (win_size - i - 1 if i+1-win_size < 0 else 0)
            observ = np.lib.pad(observ, (pad, 0), 'constant', constant_values=(0, 0))
            windowed_data.append(observ)
    else:
        for i in range(0, len(scaled_data)):
            observ = scaled_data[win_size*i:win_size*(i+1)]
            # Prevent unmatched example size and example number is much lower than data length without overlapping
            if len(observ) < win_size:
                break
            windowed_data.append(observ)
    return windowed_data

def split(data_list, split_factor=0.8):
    """Split the data"""
    length = len(data_list)
    train = data_list[0:int(split_factor*length)]
    test = data_list[int(split_factor*length):]
    return train, test

def concatenating(data_array):
    data_concatenated = [[] for i in range(len(data_array))]
    for num in range(len(data_array)):
        for i in range(len(data_array[num])):
            data_concatenated[num].extend(data_array[num][i])
    data_concatenated = np.array(data_concatenated)
    return data_concatenated


