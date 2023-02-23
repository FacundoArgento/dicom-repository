import os
from tqdm import tqdm
from pathlib import Path
from utils import get_files
from scipy.io import loadmat
from utils import create_path
from model.main import load_model
from data.extractor import ES_RANGE
from tools.metrics import Metronome
from data.store import save_csv_file
from postprocessor import postprocess
from data.extractor import get_images
from utils import filter_by_extension
from model.params import parameters as p
from model.worker import preprocess_image
from tools.calculator import compute_volumes
from data.extractor import get_parameters_info
from tools.calculator import compute_parameters
from data.extractor import get_localization_info
from tools.calculator import process_original_info

def make_predictions(image_phase_data, apply_postprocessing=False):
    ''' Run network pediction over images and store the result in the same input dictionary.'''

    bar_2 = tqdm(image_phase_data, total=len(image_phase_data), leave=False)
    for data in bar_2:
        bar_2.set_description(f'{data.name}: run prediction.')
        image, _ = preprocess_image(data.image, p['shape'], add_batch_dim=True)
        predicted = model.predict(image)
        if apply_postprocessing: predicted = postprocess(predicted)
        data.add_mask(predicted)

if __name__ == '__main__':

    import argparse

    # region Parser data
    parser = argparse.ArgumentParser(description=''' Compute clinical metrics from .mat files. ''',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--weights', required=True,
                        help='''
                        The model weights path to be tested over images. By default no models are loaded.
                        If no weights path are introduced then this scrip does\'t make anything.''')
    parser.add_argument("--data", required=False, default=None,
                        help='''
                        The path were the data are stored. The data accepted by this script are .mat files. 
                        If no .mat files are finded then this scrip doesn\'t make anything.''')
    parser.add_argument('--save', required=False, default=None,
                        help='''
                        Use --save to indicate the path where the metrics generated are saved.
                        By default this metrics are stored next to her original .mat files.''')
    parser.add_argument('--postprocess', required=False, action='store_true',
                        help='''
                        Use --postprocess if you want apply postprocessing in the predicted mask. ''')
    parser.add_argument('--summary', required=False, action='store_false',
                        help='''
                        Use --summary if you would only save the summary of the metrics collected.
                        By default all metrics are saved.''')
    # endregion Parser data
    
    # Read arguments
    args = parser.parse_args()

    if args.weights and args.data:
        
        # Metronome for store the pred stats 
        metronome = Metronome()
        
        # Create save path if this doesn't exist
        if args.save: 
            args.save = os.path.realpath(args.save)
            create_path(args.save)
        
        model = load_model(args.weights)
    
        if model:

            args.data = os.path.realpath(args.data)
            files = get_files(args.data)
            files = filter_by_extension(files, 'mat')
            
            bar = tqdm(files, total=len(files))
            for file_mat in bar:
                directory = args.save if args.save else str(Path(file_mat).parent)
                name = file_mat.split(os.sep)[-1].split('.')[0]
                case_number = name.lower().split('o')[-1]
                
                bar.set_description(f'{name}: start process.')
                data = loadmat(file_mat)
                data_info, data_image = data['setstruct'], data['im']
            
                bar.set_description(f'{name}: collecting information to make predictions.')
                info = get_parameters_info(data_info)
                time_frames, ed_slices, es_slices = get_localization_info(data_info)

                if time_frames:
                    images_phase = {}
                    es_tf = [t for t in time_frames if t in ES_RANGE]
                    ed_tf = list(set(time_frames) - set(es_tf))

                    bar.set_description(f'{name}: collecting {len(ed_slices)} ED images.')
                    images_phase['ED'] = get_images(data_image, ed_slices, ed_tf[0], case_number, 
                                                    info['ResolutionX'], info['ResolutionY'])
                    bar.set_description(f'{name}: collecting {len(es_slices)} ES images.')
                    images_phase['ES'] = get_images(data_image, es_slices, es_tf[0], case_number,
                                                    info['ResolutionX'], info['ResolutionY'])

                    bar.set_description(f'{name}: make predictions over {len(ed_slices)} ED images.')
                    make_predictions(images_phase['ED'], args.postprocess)
                    bar.set_description(f'{name}: make predictions over {len(es_slices)} ES images.')
                    make_predictions(images_phase['ES'], args.postprocess)

                    bar.set_description(f'{name}: calculating ED volume.')
                    images_phase['EDV'] = compute_volumes(images_phase['ED'], info)
                    bar.set_description(f'{name}: calculating ES volume.')
                    images_phase['ESV'] = compute_volumes(images_phase['ES'], info)

                    bar.set_description(f'{name}: calculating others parameters.')
                    params = compute_parameters(images_phase['EDV'], images_phase['ESV'], info)
                    
                    bar.set_description(f'{name}: store metrics.')
                    metronome.add_metric('Caso', int(case_number))
                    for k in params.keys(): metronome.add_metric(k, params[k])
            
            # save mean and std of the metrics collected
            save = args.save if args.save else args.data
            if args.summary: metronome.save_metrics(save, f'{model.name}.csv')
            metronome.save_statistics(save, f'Summary [{len(files)} files] ({model.name}).csv')
        