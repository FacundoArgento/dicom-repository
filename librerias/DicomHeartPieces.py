import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

class Heart_Piece():
  
    def __init__(self, time_frames, points, cavity_type):
        assert len(time_frames) == len(points)

        self.time_frames = time_frames
        self.points = points
        self.cavity_type = cavity_type

    def get_info(self):
        return zip(self.time_frames, self.points)

    def get_type(self):
        return self.cavity_type

def build_heart_pieces_from_data(data, ctype, resX, resY):
    dataX = data[0]
    dataY = data[1]
    
    cant_points, frames, slices = dataX.shape

    d = {}
    # For every slice
    for s in range(1, slices):
        slice_groupX = dataX[:,:,s]
        slice_groupY = dataY[:,:,s]
        time_frames = []; points = []

        #make all 'nan' to 0
        slice_groupX[np.isnan(slice_groupX)]=0
        slice_groupY[np.isnan(slice_groupY)]=0

        # if is empty then the slice is not useful
        if (slice_groupX.size or slice_groupY.size):
            time_frames = []
            
            # multiply the points by their resolution
            slice_groupX *= resX
            slice_groupY *= resY

            # look for the frames whit the poligons and iterate..
            for tf in range(frames):
                if slice_groupX[0,tf] != 0:
                    points_group = [(slice_groupY[i][tf], slice_groupX[i][tf]) for i in range(0, cant_points)]
                    time_frames.append(tf), points.append(points_group)
                        
            d[s+1] =  Heart_Piece(time_frames, points, ctype)
    return d

def points_to_mask(points, width, height, fill=255):

    mask = Image.new('L', (width, height))
    ImageDraw.Draw(mask).polygon(points, fill=fill, outline=None)

    return mask

def color_splash(image, mask, color, isTiff=False):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # import tifffile as tff

    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    # gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

    image_color = (np.ones(image.shape) * color).astype(np.uint8)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        # mask = (np.sum(mask, -1, keepdims=True) >= 1)
        m = Image.fromarray(mask).convert('L')
        # m.show()
        # m.save('mask.png')
        # regions = measure.regionprops(np.array(m), coordinates='rc')
        alpha_mask = np.multiply(image_color[:,:,-1], mask)
        mask = Image.fromarray(alpha_mask).convert('L')
        # mask.show()
        formatted = (image[:,:,:3] * 255 / np.max(image[:,:,:3])).astype('uint8') if (isTiff) else image[:,:,:3]
        # splash = Image.composite(Image.fromarray(image_color[:,:,:3]), Image.fromarray(image[:,:,:3]), mask)
        splash = Image.composite(Image.fromarray(image_color[:,:,:3]), Image.fromarray(formatted), mask)
        # splash.show()
    else:
        splash = Image.fromarray(image.astype(np.uint8))
    return np.array(splash)

def get_image(data, time_frame, s, resolutionX, resolutionY):
  
    # width - height - time frame - slice
    data = data[:,:, time_frame, s]
    m = data.min(); M = data.max()
    data = (((data - m) / (M - m)) * 255).astype('uint8')

    img = Image.fromarray(data)
    w, h = img.size

    return img.resize((round(w*resolutionX), round(h*resolutionY)))

def create_masks(save_path, case_folder, data_image, heart_dict, 
                 resolutionX, resolutionY, fill=255):
    
    # Number case could be in formats: caso 01 or CASO001
    # case_number = case_folder.split(' ')[-1]
    case_number = case_folder.lower().split('o')[-1].strip()

    # For every slice
    for s in heart_dict.keys():
        heart_piece = heart_dict.get(s)
        # Draw the mask and save their next to her image
        for (tf, p) in heart_piece.get_info():
            parent_path = os.path.join(save_path, case_folder, str(tf), str(s))
            Path(parent_path).mkdir(parents=True, exist_ok=True)

            # Create path for save the images
            image_path = os.path.join(parent_path, f'C{case_number}TF{tf}S{s}I.png')
            mask_name = f'C{case_number}TF{tf}S{s}{heart_piece.get_type()}.png'
            mask_path = os.path.join(parent_path, mask_name)
            
            # Build images
            img = get_image(data_image, tf, (s-1), resolutionX, resolutionY)
            w, h = img.size
            mask = points_to_mask(p, w, h, fill)

            # Save images
            if (not os.path.isfile(image_path)):
                img.save(image_path, 'PNG')
            mask.save(mask_path, 'PNG')


def build_mask(lv_mask=None, m_mask=None, rv_mask=None, 
          lv_color=85, m_color=170, rv_color=255):

    m2bool = lambda m : np.array(m, dtype=np.bool)
    bool2m = lambda m : m.astype(np.uint8)

    if (lv_mask is not None) or (m_mask is not None) or (rv_mask is not None):
        
        lv = 0 if (lv_mask is None) else bool2m(m2bool(lv_mask) * lv_color)

        m = 0 if (m_mask is None) else \
            bool2m((m2bool(lv_mask) ^ m2bool(m_mask)) * m_color) \
            if (lv_mask is not None) else \
                bool2m(m2bool(m_mask) * m_color)
        
        rv = 0 if (rv_mask is None) else \
            bool2m((m2bool(rv_mask) ^ (m2bool(m_mask) & m2bool(rv_mask))) * rv_color) \
            if (m_mask is not None) else \
                bool2m(m2bool(rv_mask) * rv_color)

        # region Debug
        # plt.imshow(lv, cmap='gray')
        # plt.show()
        # print(f'LV unique values: {np.unique(lv)}')
        # plt.imshow(m, cmap='gray')
        # plt.show()
        # print(f'M unique values: {np.unique(m)}')
        # plt.imshow(rv, cmap='gray')
        # plt.show()
        # print(f'RV unique values: {np.unique(rv)}')
        
        # i = Image.fromarray((lv + m) + rv)
        # plt.imshow(np.array(i), cmap='gray')
        # plt.show()
        # print(f'Final mask unique values: {np.unique(np.array(i))}')
        # endregion Debug

        return Image.fromarray(lv + m + rv)
    else: return None
