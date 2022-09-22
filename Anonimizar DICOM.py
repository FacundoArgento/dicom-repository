from os import fsdecode, fsencode, listdir
from os import path as pathh
from sys import exc_info
from pydicom import dcmread, dcmwrite 
from tkinter import Label
from tkinter import ttk
from tkinter import filedialog
from tkinter import Tk
import glob


def anonymize_files(path):
    recursive_path= path + "/**/*.dcm"
    i=0
    try:
        # for file in listdir(directory):
        #     filename = fsdecode(file)
        for filename in glob.glob(recursive_path, recursive=True):
            # if(filename.endswith(".dcm")):
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
            dicom.InstitutionDepartmentName=""

            dicom.PerformedProcedureStepStartDate=""
            dicom.PerformedProcedureStepID=""
            dicom.InstanceCreationDate=""
            dicom.InstanceCreationTime=""

            dicom.OtherPatientsIDs=""
            dicom.PerformingPhysicianName=""
            dicom.ReferringPhysicianName=""
    
            dcmwrite(dataset=dicom, filename=dicom_path)

            i+=1
            # if (i==20):
            #      break
    except:
        print("Exception", exc_info()[0], "occurred.")
    finally:
        dcmwrite(dataset=dicom, filename=dicom_path)


    Label(win, text="{0} Files anonymized.".format(i), font=13, background="lightblue").pack()
    button.configure(text="Select another folder")
    
def select_file():
    Label(win, text="-------------------------------------------------------------------------------------------------------------------------------------------", background="lightblue",  font=13).pack()
    Label(win, text="Anonimizing folder...", font=13, background="lightblue").pack()
    path = filedialog.askdirectory(title="Folder")
    Label(win, text="Dataset: {0}".format(path), font=13, background="lightblue").pack()
    anonymize_files(path)

#Create an instance of Tkinter frame
win= Tk()

win.geometry("1280x720")
win.resizable(0,0)
win.title("Dicom Anonimizer")
win.config(background="lightblue")

Label(win, text="Select the folder containing dicom files '.dcm'", font=('Aerial 25 bold'), background="lightblue", underline = True).pack(pady=20)

button = ttk.Button(win, text="Select folder", command= select_file)
button.pack(ipadx=5, pady=15)

Label(win, text="(Please do not close the window while performing anonimization, single files may corrupt.)", font=('Aerial 10 bold'), background="lightblue").pack()

win.mainloop()