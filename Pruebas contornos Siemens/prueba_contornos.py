import pydicom as dcm
from pydicom.overlays.numpy_handler import get_overlay_array
from pydicom.pixel_data_handlers.util import apply_voi_lut
import dicom_contour.contour as dicomContour
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import cv2 as cv


if __name__ == '__main__':

    # dicoms 4 y 5 tienen contornos......

    DICOM_PATH = "dicom-repository/Pruebas contornos Siemens/Muestra Contorno Siemens/20230203-0007/SE0100 ARGUS_-_20230203_19-35/OUTIM0004.dcm"
    OVERLAY_DATA_GROUP = 0x6000

    dicom = dcm.read_file(DICOM_PATH)

    # The numpy handler supports the conversion of data in the (60xx,3000) Overlay Data element to a ndarray 
    contour = get_overlay_array(dicom, OVERLAY_DATA_GROUP)

    plt.imshow(contour, "gray")
    plt.show()

    contour_points = cv.findContours(contour, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_points = contour_points[0] if len(contour_points) == 2 else contour_points[1]
    
    for c in contour_points:
        cv.drawContours(contour, contours=[c], contourIdx=0  , color=(255,255,255), thickness=cv.FILLED)
    
    plt.imshow(contour, "gray")
    plt.show()

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9,9))
    
    # elimina pequeños objetos de la escena (marcas de resonador para indicar que tipo de imagen se trata)
    contour = cv.morphologyEx(contour, cv.MORPH_OPEN, kernel, iterations=1)

    plt.imshow(contour,"gray")
    plt.show()