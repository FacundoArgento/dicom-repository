from curses import def_prog_mode
import os
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from PIL import Image, ImageDraw
import glob

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
            

if __name__ == '__main__':

    import argparse

    ROOT_DIRECTORY = "/home/facundo/Documents/Unnoba/Investigaciónes Patológicas/Mascaras/Mascaras Eje Corto/Ejes Cortos"
    SAVE_DIRECTORY = "/home/facundo/Documents/Unnoba/Investigaciónes Patológicas/Mascaras/Mascaras Eje Corto/Masks"

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Parse .xls notations file from segment software to binary mask.')
    parser.add_argument("--root_directory", required=False,
                        help='Root directory where the data are stored', default=ROOT_DIRECTORY)
    parser.add_argument('--save_directory', required=False,
                        metavar="/path/to/save/generate/mask/",
                        help='Directory for save the mask results', default=SAVE_DIRECTORY)
   
    args = parser.parse_args()

    folder = args.root_directory
    files_info = {}
    data_image = []

    recursive_path = folder + "/**"
    for file in glob.iglob(recursive_path, recursive=True):
    #for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        
        if(file_path.endswith(".mat")):
            data_struct = loadmat(file_path)['setstruct']
        else:
            continue
        # get x_resolution and y_resolution, [0,0][0][0]
        resX, resY = data_struct['ResolutionX'][0,0][0][0], data_struct['ResolutionY'][0,0][0][0]

        data_image = data_struct['IM'][0,0]

        # search where the data is stored...
        for i in range(data_struct['EndoX'].shape[1]):
            if data_struct['EndoX'][0,i].shape != (0,0):
                endoX = data_struct['EndoX'][0,i]
                endoY = data_struct['EndoY'][0,i]

                epiX = data_struct['EpiX'][0,i]
                epiY = data_struct['EpiY'][0,i]

                rvendoX = data_struct['RVEndoX'][0,i]
                rvendoY = data_struct['RVEndoY'][0,i]
                break
            else:
                continue

        files_info={
            'lvepi': [epiX, epiY],
            'rvendo': [rvendoX, rvendoY], 
            'lvendo':[endoX, endoY]
            }

        # folders name
        folder_name = file_path.split('.')[0].split('/')[-1]

        for part in files_info:
            part_info = build_heart_pieces_from_data(files_info.get(part), part, resX, resY)
            create_masks(args.save_directory, folder_name, data_image, part_info, resX, resY)



