import os
import numpy as np
from PIL import Image
from .params import parameters as p


# pixels values => class numbers
pixels2classes = lambda i : (i * ((p['num_classes'] - 1) / 255)).astype(np.uint8)

# class numbers => pixels values
classes2pixels = lambda i : (i * (255 / (p['num_classes'] - 1))).astype(np.uint8)

# Return a class name (cn) index
class_index = lambda cn : list(p['class_names'].keys())[list(p['class_names'].values()).index(cn)]

# Filter an image i by a class c
filter_class = lambda i, c, t=True, f=False : np.where((i == c), t, f)


def split_class_mask(mask):
  ''' Split an array class mask in their individual classes. '''
  
  # LV mask
  mask_lv = filter_class(mask, class_index('LV'))

  # M mask
  mask_m = filter_class(mask, class_index('M'))

  # RV mask
  mask_rv = filter_class(mask, class_index('RV'))

  return mask_lv, mask_m, mask_rv

def preprocess_image(img, size=p['shape'], add_batch_dim=False):
    ''' Apply pre-processing to an image. '''
    
    img = img.resize(size, Image.BICUBIC)                 # Scale
    nimg = (np.array(img) / 255.0).astype(np.float32)     # Normalize
    if add_batch_dim: nimg = np.expand_dims(nimg, axis=0) # add batch dim (1, 256, 256)

    return nimg, np.array(img)

def read_image(path, size=p['shape'], add_batch_dim=False):
    ''' Read and pre-process the image. '''
    
    img = Image.open(path) # Read

    return preprocess_image(img, size, add_batch_dim)