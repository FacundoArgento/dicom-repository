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

def obtener_punto_equivalente(event, coord_x, coord_y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(imagen1_copy, (coord_x, coord_y), 3, (0,0,255), -1)
        
        coordenada_espacio = dcm_1.coordenadas_pixel_en_espacio([coord_x,coord_y])

        # tenemos que encontrar la proyeccion de la coordenada en el espacio perteneciente al plano 1, en el plano 2.
        # La proyección de un punto P sobre un plano es otro punto P’, que está contenido en el plano
        # y es la intersección del plano con la recta que pasa por el punto P y es perpendicular el plano..

        A,B,C,D = dcm_2.get_escalares_plano()

        plano = lambda x,y,z: ((A*x) + (B*y) + (C*z) + D)
        recta = lambda t:  [(A*t) + coordenada_espacio[0], (B*t) + coordenada_espacio[1], (C*t) + coordenada_espacio[2]]
        
        t = sy.symbols("t")
        
        parte_1 = A*((A*t) + coordenada_espacio[0])
        parte_2 = B*((B*t) + coordenada_espacio[1])
        parte_3 = C*((C*t) + coordenada_espacio[2])
        
        
        solucion = sy.solve_linear(parte_1 + parte_2 + parte_3 + D, t)
        
        valor_t = np.float64(solucion[1])

        coordenada_3d_correspondiente = recta(valor_t) 

        # Verificamos que pertenezca al plano

        a = plano(coordenada_3d_correspondiente[0],coordenada_3d_correspondiente[1],coordenada_3d_correspondiente[2])
        if(a != 0):
            print(pixel_correspondiente)
            print("EL PUNTO CALCULADO NO PERTENECE AL PLANO")

        pixel_correspondiente = dcm_2.coordenadas_espacio_a_pixel(coordenada_3d_correspondiente)
        print(pixel_correspondiente)
    
        cv2.circle(imagen2_copy, pixel_correspondiente, 3, (255,0,0), -1)
        cv2.imshow("ventana 1", imagen1_copy)
        cv2.imshow("ventana 2", imagen2_copy)

    
# def obtener_punto_equivalente2(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(imagen2_copy, (x,y), 3, (0,0,255), -1)
        
#         #punto_equivalente = 

#         cv2.circle(imagen1_copy, (x,y), 3, (255,0,0), -1)
#         cv2.imshow("ventana 2", imagen2_copy)
#         cv2.imshow("ventana 1", imagen1_copy)
    


filename = 'CasoA/SE0018 CINE_TF2D13_RETRO_EJE_CORTO/OUTIM0023.dcm'
dcm_1 = Dicom(filename)

filename2 = 'CasoA/SE0004 CINE_TF2D13_RETRO_EJE_LARGO/OUTIM0001.dcm'
dcm_2= Dicom(filename2)

imagen1 = ci.leer_imagen_dicom(filename)
imagen1_copy = imagen1.copy()
#imagen1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)

imagen2 = ci.leer_imagen_dicom(filename2)
imagen2_copy = imagen2.copy()
#imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)

Fx, Fy = dcm_1.get_recta_interseccion_imagenes_Fx_Fy(dcm_2)

cv2.namedWindow("ventana 1",cv2.WINDOW_NORMAL)
cv2.namedWindow("ventana 2",cv2.WINDOW_NORMAL)

cv2.setMouseCallback('ventana 1', obtener_punto_equivalente)

#cv2.setMouseCallback('ventana 2', obtener_punto_equivalente2)

cv2.imshow("ventana 1", imagen1_copy)
cv2.imshow("ventana 2", imagen2_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()