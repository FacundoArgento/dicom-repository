import os
import numpy as np
from glob import iglob
from math import hypot
from pathlib import Path


# Create a path if this doesn't exist
create_path = lambda p : Path(p).mkdir(parents=True, exist_ok=True)

# Get all files in a directory
get_files = lambda d: [f for f in iglob(os.path.join(d, '**/*'), recursive=True) if os.path.isfile(f)]

# Filter a list of files by an extension
filter_by_extension = lambda fs, ext : list(filter(lambda f: f.split(os.sep)[-1].split('.')[-1] == ext, fs))

# Apply transformation over the pixels of the image (carry them between [0, 255])
remap = lambda i, e=1e-8 : (((i - i.min()) / (i.max() - i.min() + e)) * 255).astype('uint8')

# Convert an int array to boolean 
m2bool = lambda m : np.array(m, dtype=np.bool)

# Convert a bool array to int 
bool2m = lambda m : np.array(m, dtype=np.uint8)

# Calculate euclidean distance between two points (p) and (q)
distance = lambda p, q : hypot(*map(lambda x,y : abs(x-y), *(p,q)))


def closest_point(p, array):
  ''' Return the closest point (q) in the (array) to point (p). '''
  
  distances = [distance(p, q) for q in array]
  return array[distances.index(min(distances))]

def probabilities(arrays):
  ''' Return a distribution probabilities given a set of array with the same length. '''
  
  add = list(map(lambda *a : round(sum(a), 2), *(arrays)))
  total_add = sum(add)
  return [round(a / total_add, 2) for a in add]

