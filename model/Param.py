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
