import cv2
import numpy as np
from statistics import stdev
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from utils import bool2m, m2bool, probabilities, distance
from model.worker import class_index, split_class_mask, filter_class


def postprocess(mask):
  ''' Post processs a mask. Mask should be an array in pixels format. '''

  filter_mask = LVThreshold(mask)
  filter_mask = MThreshold(filter_mask)
  filter_mask = RVThreshold(filter_mask)

  return filter_mask


def Reconstruct(mask, num_class):
  ''' Reconstruct a predicted class if there are 
  splitted pieces and they are close each other.
  Mask must be presented in class labels. '''
  
  mask_class = filter_class(mask, num_class)
  contours, _ = cv2.findContours(bool2m(mask_class), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  
  points = []
  if len(contours) > 1:
    for c in contours: points.extend([p for p in c.tolist()])

    hull = cv2.convexHull(np.array(points))

    drawing = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(drawing, [hull], -1, 255, 1, 8)

    mask_class = binary_fill_holes(drawing)
  
  return mask_class

def CenterThreshold(mask):
    ''' If have two objects from the same 
    class only leave the nearest to the center.
    Mask must be an bool array. '''

    # calculate how many pieces are in this mask
    mask_label = label(bool2m(mask))
    props = regionprops(mask_label)
    
    if len(props) > 1:
      # center point for the cloud of points.
      centroids = [list(map(int, p.centroid[::-1])) for p in props]

      center_image = [size / 2 for size in mask.shape]

      # distance from every piece to center of the image
      distances = [distance(p, center_image) for p in centroids]

      # coordinates for every piece
      for i, pixels in enumerate([p.coords for p in props]):
        if i != distances.index(min(distances)):
          for p in pixels: mask[p[0], p[1]] = False
      
    return mask

def LVThreshold(mask):
  ''' Apply Central thresholding function to LV object. 
  Mask must be an array presented in classes labels. '''

  # split the mask in their components
  mask_lv, mask_m, mask_rv = split_class_mask(mask)

  mask_lv = CenterThreshold(mask_lv)
    
  mask = bool2m(mask_lv) * class_index('LV') | mask_m * class_index('M') | mask_rv * class_index('RV')
  
  return mask

def MThreshold(mask):
  ''' Apply thresholding functions to M object.
  Mask must be an array presented in classes labels. '''

  mask = MFilters(mask)
  
  # region CenterThreshold
  # split the mask in their components
  mask_lv, mask_m, mask_rv = split_class_mask(mask)

  mask_m = CenterThreshold(mask_m)
    
  mask = bool2m(mask_lv) * class_index('LV') | mask_m * class_index('M') | mask_rv * class_index('RV')
  # endregion CenterThreshold
  
  return mask

def MFilters(mask):
  ''' Apply filters to M object. 
  Mask must be an array presented in classes labels. '''

  # split the mask in their components
  mask_lv, mask_m, mask_rv = split_class_mask(mask)

  # calculate how many pieces are in this mask
  label_lv, label_m = label(bool2m(mask_lv)), label(bool2m(mask_m))
  props_lv, props_m = regionprops(label_lv), regionprops(label_m)

  if len(props_lv) > 1 and len(props_m) > 1:
    # region Properties
    # matrix distances from LV to M mask
    dM = distance_transform_edt(~mask_lv)
    m_distances = dM * mask_m

    # properties for M distance mask (distances + areas)
    unique_distances = np.unique(m_distances)[1::]
    m_slice_distances = np.array(list(map(lambda x : round(x, 2), unique_distances)))
    # endregion Properties
      
    # region Extract features
    features = []
    # distance between slices feature
    distances_between_slices = np.array(list(map(lambda a, b : abs(a-b), *(m_slice_distances[:-1], m_slice_distances[1:]))))
    total_diff_distances = sum(distances_between_slices)
    diff_distances_weights = np.array([d / total_diff_distances for d in distances_between_slices])
    features.append(diff_distances_weights)
    
    # pixels per area feature
    pxs_per_m_slice = np.array([np.count_nonzero(np.where(m_distances == d, 1, 0)) for d in unique_distances])
    total_px = sum(pxs_per_m_slice)
    px_weights = np.array([px / total_px for px in pxs_per_m_slice])
    features.append(px_weights)
    # endregion Extract features

    # region Process features
    # criteria for remove M pixels
    features = probabilities(tuple(features))
    in_distance = np.array(features) > stdev(features)
    # endregion Process features
    
    # region Do action
    if not all(in_distance):
      # region Threshold
      # thresholding distance
      distance_field_index, l = 0, 0
      while not distance_field_index and (l < len(in_distance) - 1):
        if not (in_distance[l] or in_distance[l + 1]): distance_field_index = l
        l += 1
      if not distance_field_index: distance_field_index = l
      cut_distance = m_slice_distances[distance_field_index]

      # threshold M mask
      cut_mask = m2bool(dM <= cut_distance) & m2bool(dM > 0)
      mask_union_m = cut_mask | mask_m

      # calculate how many M pieces are in this mask
      label_m = label(bool2m(mask_union_m))
      props_m = regionprops(label_m)

      if len(props_m) > 1: cut_mask = CenterThreshold(mask_union_m)
      # endregion Threshold

      # region Reconstruct
      # M mask for reconstruct
      mask_for_reconstruct = bool2m(cut_mask & mask_m) * class_index('M')

      # reconstruct M mask
      reconstruct_mask = Reconstruct(mask_for_reconstruct, class_index('M'))
      reconstruct_mask = (reconstruct_mask | mask_lv) ^ mask_lv
      reconstruct_mask = (reconstruct_mask | mask_rv) ^ mask_rv

      mask = mask_lv * class_index('LV') | bool2m(reconstruct_mask) * class_index('M') | mask_rv * class_index('RV')
      # endregion Reconstruct
    # endregion Do action

  return mask

def RVThreshold(mask):
  ''' Apply threshold function to RV object.
  Mask must be an array presented in classes labels. '''
  return RVReconstruct(RVFilters(mask))

def RVFilters(mask):    
  ''' Apply filters to RV object. 
  Mask must be an array presented in classes labels. '''

  points = []
  # split the mask in their components
  mask_lv, mask_m, mask_rv = split_class_mask(mask)

  # calculate how many RV pieces are in this mask
  mask_rv_int = bool2m(mask_rv)
  props_rv = regionprops(label(mask_rv_int))
  
  if len(props_rv) > 1:
    # region Properties
    # properties of every RV piece
    centroids_rv = [list(map(int, p.centroid[::-1])) for p in props_rv]
    areas_rv = [p.filled_area for p in props_rv]

    # common centroid for all RV pieces
    contours_rv, _ = cv2.findContours(mask_rv_int, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_rv: points.extend([p for p in c.tolist()])
    hull = np.squeeze(cv2.convexHull(np.array(points)))
    common_rv_centroid = hull.mean(0).tolist()

    # properties for reference mask (LV + M)
    have_substract_mask = False
    substract_mask = mask_lv | mask_m
    if substract_mask.any():
      props_sub = regionprops(bool2m(substract_mask))[0]
      centroid = list(map(int, props_sub.centroid[::-1]))
      have_substract_mask = True
    # endregion Properties

    # region Extract features
    features = []
    
    # weight by common centroid distance (Give more relevance to the RV pieces nearest to the commond centroid)
    distances = [distance(p, common_rv_centroid) for p in centroids_rv]
    total_distances = sum(distances)
    dm, ds = np.mean(distances), stdev(distances)
    penalize_distance_map = (np.array(distances) > dm) | (np.array(distances) < (dm - ds))

    # weight by areas (Give more relevance to the RV pieces with more area)
    total_area = sum(areas_rv)
    area_weights = [a / total_area for a in areas_rv]
    features.append(area_weights)
    
    # weight by (LV + M) distance (Give more relevance to the RV pieces nearest to the (LV + M) centroid)
    if have_substract_mask:
      rv_distances_to_lv_centroid = [distance(crv, centroid) for crv in centroids_rv]
      closest_rv_to_lv = rv_distances_to_lv_centroid.index(min(rv_distances_to_lv_centroid))
      penalize_distance_map[closest_rv_to_lv] = False # not penalize the most closest area to the (LV + M) centroid
      total_distance_to_lv_centroid = sum(rv_distances_to_lv_centroid)
      lv_distance_weights = [d / total_distance_to_lv_centroid for d in rv_distances_to_lv_centroid][::-1]
      features.append(lv_distance_weights)

    # penalized weights for common RV centroid
    if any(penalize_distance_map):
      penalized_amount = 2 * mask.shape[0]
      distances = list(map(lambda d, c : d + penalized_amount if c else d, *(distances, penalize_distance_map)))
      total_distances = sum(distances)
    distances_weights = [d / total_distances for d in distances][::-1]
    features.append(distances_weights)
    # endregion Extract features

    # region Process features
    # features for filter RV pieces
    features = probabilities(tuple(features))
    out_of_distance = np.array(features) < stdev(features)
    # endregion Process features

    # region Do action
    # if are any RV piece out of distance then...
    if any(out_of_distance):
      pixels_to_delete = [p for p, d in zip([p.coords for p in props_rv], out_of_distance) if d]
      # that piece are been deleted
      for pixels in pixels_to_delete:
        for p in pixels: mask_rv[p[0], p[1]] = False
      
      mask = mask_lv * class_index('LV') | mask_m * class_index('M') | bool2m(mask_rv) * class_index('RV')
    # endregion Do action
    
  return mask

def RVReconstruct(mask):
  ''' Reconstruct RV object. Mask must be an array presented in class labels. '''

  # split the mask in their components
  mask_lv, mask_m, mask_rv = split_class_mask(mask)

  reconstructed_rv = Reconstruct(mask, class_index('RV'))

  # if reconstructed mask is different from original mask
  if not (mask_rv == reconstructed_rv).all():
    substract_mask = mask_lv | mask_m

    # delete LV + M objects from reconstructed RV
    reconstructed_rv = reconstructed_rv ^ (reconstructed_rv & substract_mask)

    # return full mask with class values
    mask = mask_lv * class_index('LV') | mask_m * class_index('M') | bool2m(reconstructed_rv) * class_index('RV')

  return mask