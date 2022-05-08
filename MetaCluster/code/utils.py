import numpy as np
import os
import torch
import pickle
import random
from math import log2
from torch.utils.data import Dataset
from copy import deepcopy

def get_params(param_list):
    params = []
    count = 0
    for param in param_list:
        if count % 2 == 0:
            value = deepcopy(param.data)
            params.append(value)
            del value
        count += 1
    return params


def get_zeros_like_params(param_list):
    zeros_like_params = []
    count = 0
    for param in param_list:
        if count % 2 == 0:
            value = deepcopy(torch.zeros_like(param.data))
            zeros_like_params.append(value)
        count += 1
    return zeros_like_params


def init_params(param_list, init_values):
    count = 0
    init_count = 0
    for param in param_list:
        if count % 2 == 0:
            param.data.copy_(init_values[init_count])
            init_count += 1
        count += 1


def init_u_mem_params(param_list, init_values, bias_term, tao):
    count = 0
    init_count = 0
    for param in param_list:
        if count % 2 == 0:
            param.data.copy_(init_values[init_count]-tao*bias_term[init_count])
            init_count += 1
        count += 1


def init_ui_mem_params(param_list, init_values):
    count = 0
    for param in param_list:
        if count % 2 == 0:
            param.data.copy_(init_values)
        count += 1