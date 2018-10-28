import numpy as np
from import_data import open_data
from sklearn.preprocessing import RobustScaler

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

class Preprocesser():
    # define the parameters as class attributes for preprocessing, which are not quite often changed
    # remark: when the cut_num is too large e.g. 50, then there will be sample with empty content, thus raising error
    cut_position = 0
    def __init__(self, raw_data, processed_data, channel_list):
        self.data = processed_data
        self.raw_data = raw_data
        self.channels = channel_list
    def save(self, path):
        np.save(path, self.data)

    def select_channel(self, channel_name):
        # list.index() method returns the first value found in the list and has linear complexity, which is no problem for a short list in this case
        index = self.channels.index(channel_name)
        return self.data[index]

    @classmethod
    def from_tdm(cls, path, channels, time_index):
        data_read = open_data(path, channels)
        data_array = cls.preprocess(data_read, time_index)
        return cls(data_read, data_array, channels)

    # remark, because the length of each file is different, so the simple slicing can not be applied to the array,
    # and the data can only be represented as a nested array, outside as 2D array(channel x file) and each element itsself is automatically set to list type(with different length)
    # use for loop direktly, put the for loop outside, then it can be applied once to all the preprocessing functions
    @staticmethod
    def preprocess(data_read, time_index):
        num_of_channels = len(data_read)
        data_list = [[] for i in range(num_of_channels)]
        for channel in range(len(data_read)):
            # select the channel
            signal_list = data_read[channel]
            signal_list = list(signal_list)
            # Filewise preprocessing
            for file_number in range(data_read.shape[1]):
                signal = signal_list[file_number]
                # cut the head and tail
                signal_cut = cut_head_tail(signal, Preprocesser.cut_position)
                if not(time_index==True and channel==0):
                    # scale the data
                    signal_scaled = rob_scal(signal_cut)
                    data_list[channel].append(signal_scaled)
                else:
                    data_list[channel].append(signal_cut)
        data_array = np.array(data_list)
        return data_array