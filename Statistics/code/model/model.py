import os
import numpy as np
from . import __init__
from .worker import filter_class
import segmentation_models as sm
from tensorflow.keras import models
from .params import parameters as p

sm.set_framework('tf.keras')

class Model:

    def __init__(self, weight_path):
        self.name = weight_path.split(os.sep)[-1].split('.')[0]
        self.__weight_path = weight_path
        self.__model = self.__load_model()

    def __load_model(self):
        # Load the model
        print(f'Loading {self.name}')
        return models.load_model(self.__weight_path, compile=False)

    def predict(self, image):
        ''' Run prediction over an image. '''

        return np.argmax(self.__model.predict(image)[0], axis=-1)

