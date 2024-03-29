from http.client import PROXY_AUTHENTICATION_REQUIRED
import pydicom
from pydicom.data import get_testdata_files
import numpy as np
import cv2
# import matplotlib.pyplot as plt
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
        coordenadas_xy_pixel2 = np.abs(coordenadas_xy_pixel)
        coordenadas_xy_pixel = coordenadas_xy_pixel

        vector = [coordenadas_xy_pixel[0],coordenadas_xy_pixel[1],1]
        punto_en_el_espacio = np.dot(matriz, vector)
        
        return punto_en_el_espacio

    def coordenadas_espacio_a_pixel(self, coordenadas_3d_pixel):
        matriz_chica = self.get_matriz_reducida_mapeo_pixel_en_el_espacio()
        matriz_inversa = np.linalg.inv(matriz_chica)
        coordenada_xy = np.array(np.dot(matriz_inversa, coordenadas_3d_pixel[0:3]))
        
        return int(np.round(coordenada_xy[0])),int(np.round(coordenada_xy[1]))

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

    def graficar_rectas_interseccion_planos(self, dcm_eje2, imagen_self, imagen_dcm2):
        
        Fx, Fy = self.get_recta_interseccion_imagenes_Fx_Fy(dcm_eje2)

        # es lo mismo poner 300 que poner 2 en muestras y extremo
        # preguntar a benja

        muestras = 300
        extremo = 300
        
        filas = self.get_filas()
        columnas = self.get_columnas()
        coordenada_inicial = self.coordenadas_pixel_en_espacio([0,0])
        coordenada_extremo = self.coordenadas_pixel_en_espacio([filas,columnas])
        
        az = coordenada_inicial[2]
        bz = coordenada_extremo[2]

        # rango_zi = np.linspace(-extremo,extremo, muestras)
        rango_zi = np.linspace(az,bz, muestras)
        
        Z1 = rango_zi[0]
        Z2 = rango_zi[extremo-1]
        
        X1 = Fx(Z1)
        X2 = Fx(Z2)
        
        Y1 = Fy(X1,Z1)
        Y2 = Fy(X2,Z2)

        coordenada_3d_1 = [X1,Y1,Z1]
        coordenada_3d_2 = [X2,Y2,Z2]

        print('Coordenadas espacio')
        print(coordenada_3d_1, coordenada_3d_2)

        # Calculamos para y=0 e y=filas
        
        
        # pixels self
        pixel_11 = self.coordenadas_espacio_a_pixel(coordenada_3d_1)
        pixel_12 = self.coordenadas_espacio_a_pixel(coordenada_3d_2)

        pendiente = (pixel_12[1]-pixel_11[1])/(pixel_12[0]-pixel_11[0])
        B = pixel_11[1]-(pendiente*pixel_11[0])
    
        X0 = int(np.round(B))
        
        pixel_11 = (0, int(X0))        
        # pixels dcm2
        pixel_21 = dcm_eje2.coordenadas_espacio_a_pixel(coordenada_3d_1)
        pixel_22 = dcm_eje2.coordenadas_espacio_a_pixel(coordenada_3d_2)

        colorLinea1 = (255,255,255) 
        grosorLinea1 = 1  

        cv2.line(imagen_self, pixel_11, pixel_12 , colorLinea1, grosorLinea1)
        cv2.line(imagen_dcm2, pixel_21, pixel_22 , colorLinea1, grosorLinea1)

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
