import os
import csv
from . import __init__
from utils import get_files
from utils import create_path
from utils import filter_by_extension


def __verify_name(filename):

    file_info = filename.split('.')
    name, ext = file_info[0], file_info[-1]
    if ext.lower() != 'csv': filename = f'{name}.csv'

    return filename

def __make_row(case_number, reader):
    write_row = {'Caso': case_number}
    
    for row in reader:
        for k in row.keys(): write_row[k] = row[k]
    
    return write_row

def save_csv_file(params, save_path, filename):

    filename = __verify_name(filename)

    if not os.path.exists(save_path): save_path = create_path(save_path)
    save_path = os.path.join(save_path, filename)

    with open(save_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(params[0].keys()))
        writer.writeheader()
        for p in params:
            writer.writerow(p)

def join_csvs(csv_directory, filename):

    csv_paths = get_files(csv_directory)
    csv_paths = filter_by_extension(csv_paths, 'csv')

    filename = __verify_name(filename)

    with open(os.path.join(csv_directory, filename), 'w', newline='') as final_csv:

        header = None
        for csv_path in csv_paths:
                
            with open(csv_path, 'r') as csv_file:
                case_number = csv_path.split(os.sep)[-1].split(' ')[1]
                reader = csv.DictReader(csv_file)
                
                # write header
                if not header: 
                    header = ['Caso'] + [h for h in reader.fieldnames]
                    writer = csv.DictWriter(final_csv, fieldnames=header)
                    writer.writeheader()
                
                # write new rows
                writer.writerow(__make_row(case_number, reader))
