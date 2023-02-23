import os
import numpy as np
from . import __init__
import tensorflow as tf
from pathlib import Path
from utils import create_path
import matplotlib.pyplot as plt
import segmentation_models as sm


def verify_path(p):
    ''' Try to load a know weights if the path provided are no valid. '''

    path = os.path.realpath(p)
    if not os.path.exists(path):
        path = os.path.join(str(Path(path).parent), 'Models/Base/weights', p)

    return path if os.path.exists(path) else None

def build_save_path(in_path, suffix, out_path=None):
    if out_path: create_path(os.path.realpath(out_path))
    
    filename, ext = in_path.split(os.sep)[-1].split('.')
    filename = filename.replace('I', 'FM')
    save_name = f'{filename} ({suffix}).{ext}'

    return os.path.join(out_path, save_name) if out_path \
        else os.path.join(str(Path(in_path).parent), save_name)


parameters = {
    'shape':       (256, 256),
    'num_classes': 4,             # Classes: 0: Background - 1: LV - 2: M - 3: RV
    'lr':          1e-4,
    'batch_size':  8,
    'epochs':      100,
    'class_indx':  [1, 2, 3],     # Classes to consider.
    'weights':     [0, 1, 1, 1],  # This weights should be equals to the number of classes
    'class_names': {
        0: 'Back', 
        1: 'LV', 
        2: 'M', 
        3: 'RV'
    }
}

seed = 42
np.random.seed = seed
tf.random.set_seed(seed)

# region Loss and Metrics
class_indexes = np.array(parameters['class_indx'])
class_weights = np.array([parameters['weights'][i] for i in class_indexes])

loss = sm.losses.JaccardLoss(class_weights=class_weights, class_indexes=class_indexes)

iou_metric = sm.metrics.IOUScore(class_weights=class_weights, class_indexes=class_indexes)
fscore_metric = sm.metrics.FScore(class_weights=class_weights, class_indexes=class_indexes)

parameters['loss'] = loss
parameters['metrics'] = [iou_metric, fscore_metric]
# endregion Loss and Metrics