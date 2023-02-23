import os
import bisect
import numpy as np
from PIL import Image
from . import __init__
from math import isnan
from utils import remap
from utils import get_files
from scipy.io import loadmat


original_space_search = ['resolutionx', 'resolutiony', 'lvm','lvv', 'xsize',
                         'ysize', 'sv', 'ef', 'rvv', 'rvsv', 'rvef', 'heartrate']

ES_RANGE = range(6,17)


class ImageData():

    def __init__(self, name, image):
        self.name = name
        self.image = image
    
    def add_mask(self, mask):
        self.mask = mask

def flatten(l):
    try:
        if isinstance(l, str):
            raise TypeError()
        return flatten(l[0])
    except:
        return l

def get_parameters_info(search_space, searched_words=None, add=True, original=False):
    ''' Return all the info associated with this .mat file necessary for calculate the parameters. 
    Parameters calculated are volumes, Stroke volume, Ejection Fraction, Cardiac output. '''

    words_to_search =  original_space_search if original else \
                      ['resolutionx', 'resolutiony', 'slicethickness', 'slicegap', 'heartrate']
    info = {}

    names  = list(search_space.dtype.fields.keys())
    values = list(search_space[0][0])

    if searched_words:
        if isinstance(searched_words, list):
            if add: words_to_search.extend(searched_words) 
            else: words_to_search = searched_words
        else: 
            if add: words_to_search.append(searched_words) 
            else: words_to_search = [searched_words]

    for name, value in zip(names, values):
        if name.lower() in words_to_search:
            if name.lower() in ['lvv', 'lvm', 'rvv']:
                ed_v = value[0][flatten(search_space["EDT"]) - 1]
                es_v = value[0][flatten(search_space["EST"]) - 1]
                info[f'ED {name}'] = 0 if np.isnan(ed_v) else ed_v
                info[f'ES {name}'] = 0 if np.isnan(es_v) else es_v
            else: 
                v = flatten(value)
                info[name] = 0 if np.isnan(v) else v
    
    return info

def __get_image_location(search_space, time_frames, ed_slices, es_slices):
    ''' Search time frames and slices where the image present segmentation. '''
    
    # coord are [time_frame, slice]
    coord = [(tf,(sl+1)) for p,tfs in enumerate(search_space) \
            for tf,sls in enumerate(tfs) for sl,v in enumerate(sls) if not isnan(v)]
    
    # Delete duplicate coords
    coord = [t for t in (set(c for c in coord))]
    
    # Build list of time frames and slices found for return
    for c in coord:
        if c[0] not in time_frames: 
            bisect.insort(time_frames, c[0])
        if c[0] in ES_RANGE:
            if c[1] not in es_slices: bisect.insort(es_slices, c[1])
        else:
            if c[1] not in ed_slices: bisect.insort(ed_slices, c[1])
    
    return time_frames, ed_slices, es_slices

def get_localization_info(search_space):
    ''' Return the time frames and slices where the image present segmentation. '''
    
    time_frames = []; ed_slices = []; es_slices = []

    # LV segmentation
    time_frames, ed_slices, es_slices = __get_image_location(search_space['EndoX'][0][0], 
                                                             time_frames, ed_slices, es_slices)
    # M segmentation
    time_frames, ed_slices, es_slices = __get_image_location(search_space['EpiX'][0][0], 
                                                             time_frames, ed_slices, es_slices)
    # RV segmentation
    time_frames, ed_slices, es_slices = __get_image_location(search_space['RVEndoX'][0][0], 
                                                             time_frames, ed_slices, es_slices)

    # print(f'Find segment in {time_frames}TF and {set(ed_slices + es_slices)}S')

    return time_frames, ed_slices, es_slices

def __get_image(data, time_frame, s, resolutionX, resolutionY):
    ''' Return the image in this coordinates. '''
  
    # width - height - time frame - slice
    data = data[:,:, time_frame, s]
    data = remap(data)

    img = Image.fromarray(data)
    w, h = img.size

    return img.resize((round(w*resolutionX), round(h*resolutionY)))

def get_images(data_img, slices, tf, case, resolutionX, resolutionY):
    ''' Return the images associated with this coordinates. '''
    
    images = []
    
    for s in slices:
        images.append(ImageData(f'C{case}TF{tf}S{s}I', 
                                __get_image(data_img, tf, (s-1), resolutionX, resolutionY)))
    
    return images


if __name__ == '__main__':

    path_mat = 'UNNOBA Dataset\\pre-process data'
    path_mat = os.path.realpath(path_mat)

    files = get_files(path_mat)

    images_per_file = {}
    for file_mat in files:
        name, ext = file_mat.split(os.sep)[-1].split('.')
        if ext == 'mat':
            case = name.lower().split('o')[-1]
            data = loadmat(file_mat)
            data_image = data['im']
            data_info = data['setstruct']
            info = get_parameters_info(data_info)
            print(f'{name} with {info} values')

            time_frames, ed_slices, es_slices = get_localization_info(data_info)

            print(f'{name} with segment in {time_frames}TF and {set(ed_slices + es_slices)}S')

            if time_frames:
                images_phase = {}
                images_phase['ED'] = get_images(data_image, ed_slices, 
                                                [t for t in time_frames if t not in ES_RANGE][0], 
                                                case, info['ResolutionX'], info['ResolutionY'])
                images_phase['ES'] = get_images(data_image, es_slices, 
                                                [t for t in time_frames if t in ES_RANGE][0], 
                                                case, info['ResolutionX'], info['ResolutionY'])
                
                images_per_file[f'{name}'] = images_phase
            
    for k in images_per_file.keys():
        print(f'{k} with {len(images_per_file[k]["ES"])} images in ES phase\
              and {len(images_per_file[k]["ED"])} images in ED phase')

