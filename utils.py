"""General utility functions"""

import json
import os
import pathlib
import numpy as np

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    # ** is unpack operator, e.g. function(name,age) can take dict = {'name':'alon', 'age':18}
    # as keyword arguments: function(**dict)
    def __init__(self, **kwargs):
        # method for constructing a class from a dictionary
        self.__dict__.update(kwargs)
        pass

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def update(cls, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            return cls(**params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

def split_data_set(data_all_files, proportion_train, proportion_validation):
    """only for offline mode, and only the verified normal data set are given"""
    split_train = int(len(data_all_files)*proportion_train)
    split_validation = int(len(data_all_files)*proportion_validation)

    # shuffle along the first axis
    data_all_files = np.random.shuffle(data_all_files)

    train_data_set = data_all_files[0:split_train]
    validation_data_set = data_all_files[split_train:split_validation]
    test_data_set = data_all_files[split_validation:]

    return train_data_set, validation_data_set, test_data_set

def save_check_points(model):
    # include the epoch in the file name. (uses `str.format`)
    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    # path.dirname method returns the highest directory of the path name
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        # Save weights, every 5-epochs.
        period=5)

    # Sort the checkpoints by modification time.
    # glob method lists files in the directory
    checkpoints = pathlib.Path(checkpoint_dir).glob("*.index")
    checkpoints = sorted(checkpoints, key=lambda cp: cp.stat().st_mtime)
    checkpoints = [cp.with_suffix('') for cp in checkpoints]
    latest = str(checkpoints[-1])
    checkpoints

    # manually save weights
    model.save_weights

    # save the entire model
    model.save('my_model.h5')

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)