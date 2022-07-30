from cmath import rect
from sklearn import datasets
from dicom import Dicom
from arrow_3D import Arrow3D
import matplotlib.pyplot as plt
import numpy as np


filename = 'CasoA/SE0013 CINE_TF2D13_RETRO_EJE_CORTO/OUTIM0023.dcm'
dcm_eje_corto = Dicom(filename)

filename = 'CasoA/SE0004 CINE_TF2D13_RETRO_EJE_LARGO/OUTIM0001.dcm'
dcm_eje_largo = Dicom(filename)

escalares_plano_eje_corto = dcm_eje_corto.get_escalares_plano()

escalares_plano_eje_largo = dcm_eje_largo.get_escalares_plano()

vector_eje_corto = escalares_plano_eje_corto[:-1]
vector_eje_largo = escalares_plano_eje_largo[:-1]

vectorial = np.cross(vector_eje_corto, vector_eje_largo)
print("Producto vectorial:", vectorial)

# obtener punto

a = [escalares_plano_eje_corto[:2],escalares_plano_eje_largo[:2]]
y = [-1*escalares_plano_eje_corto[-1],-1*escalares_plano_eje_largo[-1]]
print(a)
print(y)

resultado= np.linalg.solve(a,y)

print(resultado)


# pixel = (192/2,155/2)

# Pxyz = dcm_eje_corto.coordenadas_pixel_en_espacio(pixel)
# pixelon = dcm_eje_corto.coordenadas_pixel_en_plano(Pxyz)
# print(Pxyz)
# print(pixelon)
# print("Vector eje corto: ",escalares_plano_eje_corto)
# print("Vector eje largo: ",escalares_plano_eje_largo)
