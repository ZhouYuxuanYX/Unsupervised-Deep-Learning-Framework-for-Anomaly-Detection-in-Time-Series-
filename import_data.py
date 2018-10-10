import os
import fnmatch
import amp_tdm_loader
import numpy as np

def open_data(path, channels):
    """
    Read from TDM/TDX files
    
    Args:
        path: where to read the data from
        channels: specifies selected channels as a list [channel_name1, channel_name2, ...];

    Returns:
        data_array: # channels x # files . Each element itself contains the signal of every file as 1D array
        num_of_files: the file number
    """
    namelist = fnmatch.filter(os.listdir(path), "*.TDM")
    data_list = [[] for i in range(len(channels))]
    for i in range(len(namelist)):
        file = amp_tdm_loader.OpenFile(os.path.join(path, namelist[i]))
    # Save each channel as 2 D array: #files * #signal length
        for index in range(0, len(channels)):
            data_list[index].append(file[channels[index]])
    data_array = np.asarray(data_list)
    return data_array

# Remark: the shape of the ndarray is just two dimensional, with each signal of one file as an element,
#  because the length of each file is different
