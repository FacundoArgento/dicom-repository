import numpy as np
from . import __init__
from operator import add
from functools import reduce
from collections import Counter
from model.worker import filter_class
from model.params import parameters as p


# Calculate the mass of the LVEPI cavity (v: volume - d: density)
mass = lambda v, d=1.05 : round(d * v)

# Ejection Fraction in percentage (v: sv/edv volume)
EF = lambda v : round(v * 100)

# Cardiac Output in l/min (v: volume - hr: heart rate)
CO = lambda v, hr : round((v / 1000) * hr, 2)


def __calculate_volume(img, mask, info):
    v = { 'LV': 0, 'M' : 0, 'RV': 0 }

    area_original = img.shape[0] * img.shape[1]
    area_mask = mask.shape[0] * mask.shape[1]

    # For every class
    for c in range(1, p['num_classes']):
        mask_class = filter_class(mask, c)

        if (c == 1): lv_mask = mask_class
        if (c == 2): mask_class = lv_mask | mask_class
        
        pixels = np.sum(mask_class)

        if pixels:

            if (area_mask != area_original):
                pixels = round((pixels * area_original) / area_mask)
        
            v[p['class_names'][c]] = int(pixels * round(info['SliceThickness'] + info['SliceGap']) / 1000)
        
    return v

def compute_volumes(image_phase_data, parameters_info):
    volumes = { 'LV': 0, 'M' : 0, 'RV': 0 }

    # data are a ImageData object
    for data in image_phase_data:

        volume = __calculate_volume(np.array(data.image), data.mask, parameters_info)
        # sum dict elements key by key
        volumes = reduce(add, map(Counter, [volumes, volume]))
        
    return volumes

def process_original_info(original_info):
    params = {}

    # Mass
    ed_lvm = mass(original_info['ED LVM'])
    es_lvm = mass(original_info['ES LVM'])

    # Ejection fraction
    lv_ef = EF(original_info['EF'])
    rv_ef = EF(original_info['RVEF'])

    # Cardiac output
    lv_co = CO(original_info['SV'], original_info['HeartRate'])

    # region Store values
    params['W'] = round(original_info['YSize']*original_info['ResolutionY'])
    params['H'] = round(original_info['XSize']*original_info['ResolutionX'])
    params['Y (mm)'] = round(original_info['ResolutionY'], 3)
    params['X (mm)'] = round(original_info['ResolutionX'], 3)

    params['ED LVM (g)'] = ed_lvm
    params['ES LVM (g)'] = es_lvm
    params['ED LVV (ml)'] = round(original_info['ED LVV'])
    params['ES LVV (ml)'] = round(original_info['ES LVV'])
    params['LV SV (ml)'] = round(original_info['SV'])
    params['LV EF (%)'] = lv_ef
    params['LV CO (l/min)'] = lv_co
    
    params['ED RVV (ml)'] = round(original_info['ED RVV'])
    params['ES RVV (ml)'] = round(original_info['ES RVV'])
    params['RV SV (ml)'] = round(original_info['RVSV'])
    params['RV EF (%)'] = rv_ef

    # endregion Store values

    return params


def compute_parameters(ed_volum, es_volum, info):
    params = {}

    # Mass
    ed_lvm = mass((ed_volum['M'] - ed_volum['LV']))
    es_lvm = mass((es_volum['M'] - es_volum['LV']))

    # Stroke volume
    lv_sv = ed_volum['LV'] - es_volum['LV']
    rv_sv = ed_volum['RV'] - es_volum['RV']

    # Ejection fraction
    lv_ef = EF(lv_sv / ed_volum['LV'])
    rv_ef = EF(rv_sv / ed_volum['RV'])

    # Cardiac output
    lv_co = CO(lv_sv, info['HeartRate'])

    # region Store values
    params['ED LVM (g)'] = ed_lvm
    params['ES LVM (g)'] = es_lvm
    params['ED LVV (ml)'] = ed_volum['LV']
    params['ES LVV (ml)'] = es_volum['LV']
    params['LV SV (ml)'] = lv_sv
    params['LV EF (%)'] = lv_ef
    params['LV CO (l/min)'] = lv_co
    
    params['ED RVV (ml)'] = ed_volum['RV']
    params['ES RVV (ml)'] = es_volum['RV']
    params['RV SV (ml)'] = rv_sv
    params['RV EF (%)'] = rv_ef
    # endregion Store values

    return params