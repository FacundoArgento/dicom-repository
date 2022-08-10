from re import I
import cv2
from cmath import rect
from cv2 import cvtColor
from sklearn import datasets
from dicom import Dicom
from arrow_3D import Arrow3D
import matplotlib.pyplot as plt
import numpy as np
import comandos_imagenes as ci
import sympy as sy
from skimage import color
from skimage import io

def obtener_punto_dcm1(event, coord_x, coord_y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        pixel = [coord_x, coord_y]
        pixel_espacio = dcm_1.coordenadas_pixel_en_espacio(pixel)

        ## Verificamos que el punto pertenezca a la recta para no hacer calculos de gusto

        # calculamos el punto equivalente en las demas fotos
        
        pixel_correspondiente_dcm2 = dcm_2.coordenadas_espacio_a_pixel(pixel_espacio)
        pixel_correspondiente_dcm3 = dcm_3.coordenadas_espacio_a_pixel(pixel_espacio)
        pixel_correspondiente_dcm4 = dcm_4.coordenadas_espacio_a_pixel(pixel_espacio)

        cv2.circle(imagen1_copy, pixel, 1, (125,125,0), 1)
        cv2.circle(imagen2_copy, pixel_correspondiente_dcm2, 1, (255,0,0), 1)
        cv2.circle(imagen3_copy, pixel_correspondiente_dcm3, 1, (0,255,0), 1)
        cv2.circle(imagen4_copy, pixel_correspondiente_dcm4, 1, (0,0,255), 1)

        cv2.imshow("ventana 1", imagen1_copy)
        cv2.imshow("ventana 2", imagen2_copy)
        cv2.imshow("ventana 3", imagen3_copy)
        cv2.imshow("ventana 4", imagen4_copy)



# Lectura de Dicoms

filename = 'CasoA/SE0018 CINE_TF2D13_RETRO_EJE_CORTO/OUTIM0023.dcm'
dcm_1 = Dicom(filename)

filename2 = 'CasoA/SE0004 CINE_TF2D13_RETRO_EJE_LARGO/OUTIM0001.dcm'
dcm_2= Dicom(filename2)

filename3 = 'CasoA/SE0025 CINE_TF2D13_RETRO_TSVI/OUTIM0011.dcm'
dcm_3 = Dicom(filename3)

filename4 = 'CasoA/SE0008 CINE_TF2D13_RETRO_4_CAMARAS/OUTIM0006.dcm'
dcm_4= Dicom(filename4)


# Sacamos una copia de las imagenes de cada dicom (las paso a rgb para graficar los puntos)

imagen1 = ci.leer_imagen_dicom(filename)
imagen1_copy = color.gray2rgb(imagen1.copy())

imagen2 = ci.leer_imagen_dicom(filename2)
imagen2_copy = color.gray2rgb(imagen2.copy())

imagen3= ci.leer_imagen_dicom(filename3)
imagen3_copy = color.gray2rgb(imagen3.copy())

imagen4 = ci.leer_imagen_dicom(filename4)
imagen4_copy = color.gray2rgb(imagen4.copy())

## WINDOWS

cv2.namedWindow("ventana 1",cv2.WINDOW_NORMAL)
cv2.namedWindow("ventana 2",cv2.WINDOW_NORMAL)
cv2.namedWindow("ventana 3",cv2.WINDOW_NORMAL)
cv2.namedWindow("ventana 4",cv2.WINDOW_NORMAL)

## Rectas interseccion con un dicom

dcm_1.graficar_rectas_interseccion_planos(dcm_2, imagen1_copy, imagen2_copy)
dcm_1.graficar_rectas_interseccion_planos(dcm_3, imagen1_copy, imagen3_copy)
dcm_1.graficar_rectas_interseccion_planos(dcm_4, imagen1_copy, imagen4_copy)

cv2.setMouseCallback('ventana 1', obtener_punto_dcm1)
# cv2.setMouseCallback('ventana 2', obtener_punto_dcm2)
# cv2.setMouseCallback('ventana 3', obtener_punto_dcm3)
# cv2.setMouseCallback('ventana 4', obtener_punto_dcm4)

#cv2.setMouseCallback('ventana 2', obtener_punto_equivalente2)

cv2.imshow("ventana 1", imagen1_copy)
cv2.imshow("ventana 2", imagen2_copy)
cv2.imshow("ventana 3", imagen3_copy)
cv2.imshow("ventana 4", imagen4_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()