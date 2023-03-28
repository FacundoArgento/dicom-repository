import os
from sys import exc_info
import numpy as np
from scipy.io import loadmat
from pydicom import dcmread, dcmwrite 
from PIL import Image
from tqdm import tqdm
from glob import iglob
from pathlib import Path
from os import path as pathh


from DicomHeartPieces import build_heart_pieces_from_data, create_masks, build_mask

def anonymize_files(path):
    """
    Anonimizes dicom files.

    :param path: Path (string) where data are stored.
    """

    recursive_path= path + "/**"
    i=0
    try:
        for filename in iglob(recursive_path, recursive=True):
            if(filename.endswith(".dcm")):
                dicom_path = pathh.join(path, filename)
                
                dicom = dcmread(dicom_path)

                dicom.AcquisitionDate=""
                dicom.AccessionNumber=""
                dicom.AdditionalPatientHistory=""
                    
                dicom.ContentDate=""
                dicom.ContentTime=""
                dicom.PatientName=""
                dicom.PatientID=""
                dicom.PatientBirthDate=""

                dicom.InstitutionName=""
                dicom.InstitutionAddress=""
                dicom.InstitutionalDepartmentName=""

                dicom.StudyID=""
                dicom.StationName=""
                dicom.StudyDate=""
                dicom.SeriesDate=""
                # Requested Procedure ID & Scheduled Procedure Step Id
                # dicom.RequestAttributesSequence._list[0].ScheduledProcedureStepID=""
                # dicom.RequestAttributesSequence._list[0].RequestedProcedureID=""
                dicom.RequestAttributesSequence=[]

                dicom.PerformedProcedureStepStartDate=""
                dicom.PerformedProcedureStepID=""
                dicom.InstanceCreationDate=""
                dicom.InstanceCreationTime=""

                dicom.OtherPatientsIDs=""
                dicom.OperatorsName=""
                dicom.PerformingPhysicianName=""
                dicom.PhysiciansOfRecord=""
                dicom.ReferringPhysicianName=""
                dicom.RequestingPhysician=""


                dcmwrite(dataset=dicom, filename=dicom_path)

                i+=1
            else:
                continue
    except FileNotFoundError as e:
        print("Exception", exc_info()[0], "occurred.")
        raise FileNotFoundError("File not found.") from e
    except ValueError as e:
        print("Exception", exc_info()[0], "occurred.")
        raise ValueError("Value not found") from e
    finally:
        if(dicom != None):
            dcmwrite(dataset=dicom, filename=dicom_path)

    print(f"{i} files anonymized successfully.")


def generate_mask_from_mat_file(root_directory, save_directory):
    """
    Generate masks from mat file by an given directory, and save the results in the directory 
    indicated in save_directory.

    :param root_directory: Root directory (string) where the data are stored.
    :param save_directory: Directory for save the mask results.
    """
    folder = root_directory
    files_info = {}
    data_image = []

    recursive_path = folder + "/**"
    for file in iglob(recursive_path, recursive=True):
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
            create_masks(save_directory, folder_name, data_image, part_info, resX, resY)



def build_full_mask(root_directory, mask_name="new mask.png", 
                   lv_color=85, m_color=170, rv_color=255):
    """
    Description here.

    :param root_directory: Root directory (string) where the data are stored.
    :param mask_name: Use this name if original image are not presented. Default 'new mask.png'
    :param Color for paint the LV strcture. Default value = 85
    :param Color for paint the M strcture. Default value = 170
    :param Color for paint the RV strcture.Default value = 255
    """

    search_path = os.path.join(root_directory, '**/*')
    files_paths = [f for f in iglob(search_path, recursive=True) if os.path.isfile(f)]
    files_paths.append(os.path.join('This', 'Fake', 'Dir', 'Allow Us', 'Save', 'The', 'Last', 'Mask'))

    # n: file_name - p: path_to_image - c: cavity_type - m: mask
    read_image = lambda n, p, c, m : Image.open(p) if (c in n) else m
    
    prev_folder_directory = None
    for file_path in tqdm(files_paths, total=len(files_paths)):
        folder_directory = str(Path(file_path).parent)
        file_name = file_path.split(os.sep)[-1].upper()

        if (prev_folder_directory != folder_directory):
            if (prev_folder_directory is not None):
                mask_path = os.path.join(prev_folder_directory, mask_name)
                full_mask = build_mask(lv_mask, m_mask, rv_mask, 
                            lv_color, m_color, rv_color)

                if (full_mask is not None): full_mask.save(mask_path, 'png')
            
            lv_mask = None; m_mask = None; rv_mask = None
            mask_name = mask_name

        if (files_paths.index(file_path) != (len(files_paths) - 1)):
            mask_name = file_name.replace('I', 'FM') if ('I' in file_name) and ('LVEPI' not in file_name) else mask_name
            lv_mask = read_image(file_name, file_path, 'LVENDO', lv_mask)
            m_mask  = read_image(file_name, file_path, 'LVEPI',  m_mask)
            rv_mask = read_image(file_name, file_path, 'RVENDO', rv_mask)
            
            prev_folder_directory = folder_directory