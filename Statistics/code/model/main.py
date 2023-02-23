import os
from PIL import Image
from tqdm import tqdm
from . import __init__
from .model import Model
from pathlib import Path
from utils import get_files
from .worker import read_image
from .params import verify_path
from .worker import classes2pixels
from .params import build_save_path
from .params import parameters as p
from postprocessor import postprocess


def load_model(weights_path):
    model_path = verify_path(weights_path)
    assert model_path, 'Fail to load that model. The path does\'t exist.'

    return Model(model_path)


if __name__ == '__main__':
    
    import argparse

    # region Parser data
    parser = argparse.ArgumentParser(description='''Create masks for an input images.''',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--weights', required=True,
                        help='''
                        The model weights path to be tested over images. By default no models are loaded.
                        If no weights path are introduced then this scrip does\'t make anything.''')
    parser.add_argument("--images", required=False, default=None,
                        help='''
                        The image to be  processed. This could be an image path or a folder with images. 
                        If no image path are introduced then this scrip does\'t make anything.''')
    parser.add_argument('--save', required=False, default=None,
                        help='''
                        Use --save to indicate the path where the masks generated are saved.
                        By default this masks are stored next to the original files.''')
    parser.add_argument('--postprocess', required=False, action='store_true',
                        help='''
                        Use --postprocess if you want apply postprocessing in the predicted mask. ''')
    # endregion Parser data
    
    # Read arguments
    args = parser.parse_args()

    # If weights are provided load that model
    if args.weights and args.images:

        model = load_model(args.weights)

        # region process images
        images_paths = [args.images] if os.path.isfile(args.images) else get_files(args.images)

        run_predictions(images_paths, model, args.save)

        print('processing images...')
        for image_path in tqdm(images_paths, total=len(images_paths)):
            nimage, image = read_image(image_path, add_batch_dim=True)
            
            # predict
            predicted = model.predict(nimage)
            if args.postprocess: predicted = classes2pixels(postprocess(predicted))
            
            save_path = build_save_path(image_path, model.name.split(".")[0], args.save)
            Image.fromarray(predicted).save(save_path)
        # endregion process images
