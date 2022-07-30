from http.client import PROXY_AUTHENTICATION_REQUIRED
import pydicom
from pydicom.data import get_testdata_files
import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import math

class Dicom:

    def __init__(self, ruta_imagen):
        self.dataset = pydicom.dcmread(ruta_imagen)
        self.delta_i = self.dataset.PixelSpacing[0]
        self.delta_j = self.dataset.PixelSpacing[1]
        # Información sacada de: https://dicom.innolitics.com/ciods/rt-dose/image-plane/00200037
        self.imagen = self.dataset.pixel_array   
        self.filas = self.dataset.Rows
        self.columnas = self.dataset.Columns

        self.orientacion = self.dataset.ImageOrientationPatient
        self.posicion = self.dataset.ImagePositionPatient
        self.__extraer_coordenadas_image_orientation__()

        print("Posicion")
        print(self.posicion)
        print("Orientacion")
        print(self.orientacion)

    def get_vector_fila_orientacion(self):
        return [self.xx, self.xy, self.xz]

    def get_vector_columna_orientacion(self):
        return [self.yx, self.yy, self.yz]

    def __extraer_coordenadas_image_orientation__(self):
        orientacion = self.get_orientacion()
        self.xx = orientacion[0]
        self.xy = orientacion[1]
        self.xz = orientacion[2]
        self.yx = orientacion[3]
        self.yy = orientacion[4]
        self.yz = orientacion[5]

    def coordenadas_espacio_a_pixel(self, coordenadas_3d_pixel):
        matriz_chica = self.get_matriz_reducida_mapeo_pixel_en_el_espacio()
        matriz_inversa = np.linalg.inv(matriz_chica)
        coordenada_xy = np.array(np.dot(matriz_inversa, coordenadas_3d_pixel[0:3]))
        
        return int(np.round(coordenada_xy[0])),int(np.round(coordenada_xy[1]))

    def get_matriz_reducida_mapeo_pixel_en_el_espacio(self):
        posicion = self.get_posicion()

        f11 = self.xx * self.delta_i
        f12 = self.yx * self.delta_j
        f14 = posicion[0]

        f21 = self.xy * self.delta_i
        f22 = self.yy * self.delta_j
        f24 = posicion[1]

        f31 = self.xz * self.delta_i
        f32 = self.yz * self.delta_j
        f34 = posicion[2]
        
        matriz = np.array([[f11, f12,f14],
                            [f21, f22, f24],
                            [f31, f32, f34]])

        return matriz

    def coordenadas_pixel_en_espacio(self, coordenadas_xy_pixel):
        matriz = self.get_matriz_reducida_mapeo_pixel_en_el_espacio()

        # vector = [coordenadas_xy_pixel[0],coordenadas_xy_pixel[1],0,1]
        coordenadas_xy_pixel = np.abs(coordenadas_xy_pixel)

        vector = [coordenadas_xy_pixel[0],coordenadas_xy_pixel[1],1]
        punto_en_el_espacio = np.dot(matriz, vector)
        
        return punto_en_el_espacio

    def coordenadas_pixel_en_plano(self, punto_en_el_espacio):

        matriz = self.get_matriz_reducida_mapeo_pixel_en_el_espacio()

        inversa = matriz.copy()
        # determinante = np.linalg.det(matriz)
        matriz_inversa = np.linalg.inv(inversa)
        punto_en_el_plano = np.array(np.dot(matriz_inversa, punto_en_el_espacio))

        punto_x = int(np.round(punto_en_el_plano[0]))
        punto_y = int(np.round(punto_en_el_plano[1]))
    
        punto_xy = (punto_x, punto_y)
        return punto_xy


    def get_ecuacion_del_plano_imagen(self):
        
        detx, dety, detz, desplazamiento = self.get_escalares_plano()

        if ((detz==0.0) or (detz==-0.0)):
            detz = 1
            A = (detx/dety)
            # A = 0
            # B = (dety/detx)
            B = 0
            C = (desplazamiento/dety)
            eje = 'y'

            plano = lambda x,y:A*x + C
        else:
            A = (detx/detz)
            B = (dety/detz)
            C = (desplazamiento/detz)

            plano = lambda x,y:-A*x + B*y - C
            eje = 'z'

        return plano, eje

    def get_recta_interseccion_imagenes_Fx_Fy(self, dcm_eje2):
        A1,B1,C1,D1 = self.get_escalares_plano()
        A2,B2,C2,D2 = dcm_eje2.get_escalares_plano()
        A = A1-A2
        B = B1-B2
        C = C1-C2
        D = D1-D2
        Af = (A1-((B1*A)/B))
        Cf = (C1-((B1*C)/B))    
        Df = (D1-((B1*D)/B))
        Fx = lambda z: -(Cf/Af)*z - (Df/Af)
        Fy = lambda x,z: (A/B)*x + (C/B)*z + (D/B)
        return Fx, Fy


    def get_escalares_plano(self):

        # Funcion para obtener los escalares del plano por separado

        u = self.get_vector_columna_orientacion()
        ux = float(u[0])
        uy = float(u[1])
        uz = float(u[2])
        # v = [-2,4,7]
        v = self.get_vector_fila_orientacion()
        vx = float(v[0])
        vy = float(v[1])
        vz = float(v[2])
        # P = [-1,2,6]
        P = self.get_posicion()
        Px = float(P[0])
        Py = float(P[1])
        Pz = float(P[2])
        # [ux, uy, uz]
        # [vx, vy, vz]
        detx = (uy*vz) - (vy*uz)
        # dety = -((ux*vz) - (vx*uz))
        dety = (ux*vz) - (vx*uz)
        detz = (ux*vy) - (uy*vx)
        #desplazamiento = -((detx*Px) + (dety*Py) + (detz*Pz))
        desplazamiento = -(detx*Px) + (dety*Py) - (detz*Pz)

        return detx, dety, detz, desplazamiento


    def get_ecuacion_del_plano_imagen_escalares(self):
       
        detx, dety, detz, desplazamiento = self.get_escalares_plano()

        print(detz)
        print(desplazamiento)
        if ((detz==0.0) or (detz==-0.0)):
            # detz = 1
            A = (detx/dety)
            # A = 0
            # B = (dety/detx)
            B = 0
            C = (desplazamiento/dety)
            eje = 'y'

            plano = lambda x,y:A*x + C
        else:
            A = (detx/detz)
            B = (dety/detz)
            C = (desplazamiento/detz)

            plano = lambda x,y:-A*x + B*y - C
            eje = 'z'

        return A,B,C,eje
    
    def get_imagen(self):
        
        return self.imagen

    def get_posicion(self):
        return self.posicion

    def get_orientacion(self):
        return self.orientacion

    def get_filas(self):
        return self.filas

    def get_columnas(self):
        return self.columnas
