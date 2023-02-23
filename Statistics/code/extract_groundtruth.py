import os
from tqdm import tqdm
from pathlib import Path
from utils import get_files
from scipy.io import loadmat
from utils import create_path
from tools.metrics import Metronome
from data.store import save_csv_file
from utils import filter_by_extension
from data.extractor import get_parameters_info
from tools.calculator import process_original_info

def get_groundtruth(data):
    
    original_info = get_parameters_info(data, original=True)
    return  process_original_info(original_info)

if __name__ == '__main__':

    import argparse

    # region Parser data
    parser = argparse.ArgumentParser(description='''Create masks for an input images.''',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", required=True,
                        help='''
                        The path were the data are stored. The data accepted by this script are .mat files. 
                        If no .mat files are finded then this scrip doesn\'t make anything.''')
    parser.add_argument('--save', required=False, default=None,
                        help='''
                        Use --save to indicate the path where the metrics generated are saved.
                        By default this metrics are stored next to her original .mat files.''')
    parser.add_argument('--summary', required=False, action='store_false',
                        help='''
                        Use --summary if you would only save the summary of the metrics collected.
                        By default all metrics are saved.''')
    # endregion Parser data
    
    # Read arguments
    args = parser.parse_args()

    args.data = os.path.realpath(args.data)
    files = get_files(args.data)
    files = filter_by_extension(files, 'mat')

    # If the data path doesn't exist or no are .mat files inside there exit
    if os.path.exists(args.data) or len(files):

        metronome = Metronome()

        # Create save path if this doesn't exist
        if args.save: 
            args.save = os.path.realpath(args.save)
            create_path(args.save)

        bar = tqdm(files, total=len(files))
        for file_mat in bar:
            directory = args.save if args.save else str(Path(file_mat).parent)
            name = file_mat.split(os.sep)[-1].split('.')[0]
            case_number = name.lower().split('o')[-1]
            bar.set_description(f'{name}: start process.')

            data = loadmat(file_mat)
            data_info = data['setstruct']
            bar.set_description(f'{name}: collecting groundtruth information.')
            params = get_groundtruth(data_info)
            
            bar.set_description(f'{name}: store metrics.')
            metronome.add_metric('Caso', int(case_number))
            for k in params.keys(): metronome.add_metric(k, params[k])
        
        # save mean and std of the metrics collected
        save = args.save if args.save else args.data
        if args.summary: metronome.save_metrics(save, f'groundtruth.csv')
        metronome.save_statistics(save, f'Summary [{len(files)} files] (groundtruth).csv', start_index=5)
