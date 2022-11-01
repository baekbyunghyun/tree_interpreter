import numpy as np


def get_str_to_integer(str):
    if (str is None) or (str == ""):
        return None

    return int(str)


def get_str_to_float(str):
    if (str is None) or (str == ""):
        return None

    return float(str)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_value_in_dict(key, value_dict):
    if key in value_dict.keys():
        return value_dict[key]

    return None


def get_value_in_dict_using_multiple_keys(key_list, value_dict):
    value = None

    for key in key_list:
        if key in value_dict.keys():
            value = value_dict[key]

    return value
