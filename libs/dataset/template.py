import torch
import os
import math
import cv2
import numpy as np

import json
import yaml
import pickle

from PIL import Image
from .data import BaseData, register_data

class CustomData(BaseData):

    def __init__(self, *args, **kwarg):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def set_max_skip(self):
        pass

    def increase_max_skip(self):
        pass

# optional dali data source
class DaliCustomData(CustomData):

    def __init__(self, *args, **kwarg):
        super(DaliCustomData, self).__init__(*args, **kwarg)
        pass

    def __getitem__(self, idx):
        pass

register_data('ALIAS', CustomData)
register_data('DALIALIAS', DaliCustomData)