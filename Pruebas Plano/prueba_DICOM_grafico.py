from re import I
import cv2
from cmath import rect
from sklearn import datasets
from dicom import Dicom
from arrow_3D import Arrow3D
import matplotlib.pyplot as plt
import numpy as np

def graficar_recta_interseccion(dcm_eje1,dcm_eje2):
    Fx, Fy = dcm_eje1.get_recta_interseccion_imagenes_Fx_Fy(dcm_eje2)
    
    muestras = 300
    rango_zi = np.linspace(-300,300, muestras)
    rango_xi = Fx(rango_zi)
    rango_yi = Fy(rango_xi, rango_zi)
     
    grafica.plot3D(rango_xi,rango_yi,rango_zi,color="black")   


def dibujar_recta_interseccion(dcm_eje1,dcm_eje2):
    Fx, Fy = dcm_eje1.get_recta_interseccion_imagenes_Fx_Fy(dcm_eje2)

    img = dcm_eje1.get_imagen().copy()
    img2 = dcm_eje2.get_imagen().copy()

    # es lo mismo poner 300 que poner 2 en muestras y extremo
    # preguntar a benja

    muestras = 300
    extremo = 300
    
    filas = dcm_eje1.get_filas()
    columnas = dcm_eje1.get_columnas()
    coordenada_inicial = dcm_eje1.coordenadas_pixel_en_espacio([0,0])
    coordenada_extremo = dcm_eje1.coordenadas_pixel_en_espacio([filas,columnas])
    # coordenada_inicial = [-200,-200]
    # coordenada_extremo = [200,200]
    
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

    pixel_11 = dcm_eje1.coordenadas_espacio_a_pixel(coordenada_3d_1)
    pixel_12 = dcm_eje1.coordenadas_espacio_a_pixel(coordenada_3d_2)
    print('Coordenadas PX 1 y 2')
    print(pixel_11, pixel_12)

    colorLinea1 = (255,0,0) 
    grosorLinea1 = 1  
    # cv2.line(img, pixel_11, pixel_12 , colorLinea1, grosorLinea1)
    plt.figure()
    plt.title("El titulo")
    cv2.line(img, pixel_11, pixel_12 , colorLinea1, grosorLinea1 )
    plt.imshow(img, cmap="gray")

    pixel_21 = dcm_eje2.coordenadas_espacio_a_pixel(coordenada_3d_1)
    pixel_22 = dcm_eje2.coordenadas_espacio_a_pixel(coordenada_3d_2)
    
    print('Coordenadas PX 3 y 4')
    print(pixel_21, pixel_22)

    plt.figure()
    plt.title("Eje Largo ")
    cv2.line(img2, pixel_21, pixel_22 , colorLinea1, grosorLinea1 )
    plt.imshow(img2, cmap="gray")

    
def graficar_plano_EL(dcm_eje, grafica, color_punto='red', color_plano='red'):
    posicion = np.array(dcm_eje.get_posicion())
    # orientacion = dcm_eje.get_orientacion()

    vector_orientacion_fila = dcm_eje.get_vector_fila_orientacion()*dcm_eje.get_filas()
    vector_orientacion_columna = dcm_eje.get_vector_columna_orientacion()*dcm_eje.get_columnas()

    z, eje = dcm_eje.get_ecuacion_del_plano_imagen()
    A,B,C, eje = dcm_eje.get_ecuacion_del_plano_imagen_escalares()
    filas = dcm_eje.get_filas()
    columnas = dcm_eje.get_columnas()
    coordenada_inicial = dcm_eje.coordenadas_pixel_en_espacio([0,0])
    coordenada_extremo = dcm_eje.coordenadas_pixel_en_espacio([filas,columnas])
    # coordenada_inicial = [-200,-200]
    # coordenada_extremo = [200,200]
    
    ax = coordenada_inicial[0]    
    ay = coordenada_inicial[1]
    az = coordenada_inicial[2]
    bx = coordenada_extremo[0]
    by = coordenada_extremo[1]
    bz = coordenada_extremo[2]
    
    muestras = 11
    # rango_xi = np.linspace(ax,bx, muestras)
    # rango_yi = np.linspace(ay,by, muestras)
    # rango_zi = np.linspace(az,bz, muestras)
    rango_xi = np.linspace(-200,200, muestras)
    rango_yi = np.linspace(-200,200, muestras)
    rango_zi = np.linspace(-200,200, muestras)
    Xi, Yi= np.meshgrid(rango_xi,rango_yi)
    
    plano_3D = z(Xi,Yi)

    # Punto solución del sistema 3x3
    grafica.scatter(posicion[0],posicion[1],posicion[2],
                    color = color_punto,
                    marker='o',
                    label ='punto',
                    linewidth = 6)

    grafica.plot_surface(Xi,plano_3D,Yi,
                        color =color_plano,
                        label='Ecuación 1')

def graficar_plano_EC(dcm_eje, grafica, color_punto='red', color_plano='red'):

    posicion = np.array(dcm_eje.get_posicion())

    vector_orientacion_fila = dcm_eje.get_vector_fila_orientacion()*dcm_eje.get_filas()
    vector_orientacion_columna = dcm_eje.get_vector_columna_orientacion()*dcm_eje.get_columnas()

    z, eje = dcm_eje.get_ecuacion_del_plano_imagen()
    A,B,C, eje = dcm_eje.get_ecuacion_del_plano_imagen_escalares()
    filas = dcm_eje.get_filas()
    columnas = dcm_eje.get_columnas()
    coordenada_inicial = dcm_eje.coordenadas_pixel_en_espacio([0,0])
    coordenada_extremo = dcm_eje.coordenadas_pixel_en_espacio([filas,columnas])
    # coordenada_inicial = [-200,-200]
    # coordenada_extremo = [200,200]
    
    ax = coordenada_inicial[0]    
    ay = coordenada_inicial[1]
    az = coordenada_inicial[2]
    bx = coordenada_extremo[0]
    by = coordenada_extremo[1]
    bz = coordenada_extremo[2]
    
    muestras = 11
    rango_xi = np.linspace(ax,bx, muestras)
    rango_yi = np.linspace(ay,by, muestras)
    rango_zi = np.linspace(az,bz, muestras)
    Xi, Yi= np.meshgrid(rango_xi,rango_yi)
    if (eje=='z'):
        Xi, Yi= np.meshgrid(rango_xi,rango_yi)
    elif (eje=='y'):
        Xi, Yi= np.meshgrid(rango_xi,rango_zi)
    else:
        Xi, Yi= np.meshgrid(rango_yi,rango_zi)

    plano_3D = z(Xi,Yi)

    # Punto solución del sistema 3x3
    grafica.scatter(posicion[0],posicion[1],posicion[2],
                    color = color_punto,
                    marker='o',
                    label ='punto',
                    linewidth = 6)

    grafica.arrow3D(0,0,0,
            vector_orientacion_fila[0],vector_orientacion_fila[1],vector_orientacion_fila[2],
            mutation_scale=20,
            ec ='green',
            fc='red')

    grafica.arrow3D(0,0,0,
            vector_orientacion_columna[0],vector_orientacion_columna[1],vector_orientacion_columna[2],
            mutation_scale=20,
            ec ='green',
            fc='red')

    if (eje=='y'):
        # grafica.plot_surface(Xi,Yi,plano_Z0,
        #                     color =color_plano,
        #                     label='Ecuación 1')
        # Zi = np.zeros((muestras,muestras))
        grafica.plot_surface(Xi,plano_3D,Yi,
                            color =color_plano,
                            label='Ecuación 1')
        print('y')
    else:
        grafica.plot_surface(Xi,Yi,plano_3D,
                            color =color_plano,
                            label='Ecuación 1')



# Comienzo

filename = 'CasoA/SE0013 CINE_TF2D13_RETRO_EJE_CORTO/OUTIM0023.dcm'
dcm_1 = Dicom(filename)

filename2 = 'CasoA/SE0004 CINE_TF2D13_RETRO_EJE_LARGO/OUTIM0001.dcm'
dcm_2= Dicom(filename2)

# GRAFICA de planos

figura = plt.figure()
grafica = figura.add_subplot(111, projection='3d')

# Calculamos la recta interseccion entre ambos planos y la graficamos

graficar_plano_EC(dcm_1, grafica, color_punto='red', color_plano='blue')
graficar_plano_EL(dcm_2, grafica, color_punto='blue', color_plano='red')
graficar_recta_interseccion(dcm_1, dcm_2)

# Dibujamos las rectas en las imagenesss

dibujar_recta_interseccion(dcm_1,dcm_2)


grafica.set_title('Grafica Interseccion entre Planos')
grafica.set_xlabel('x')
grafica.set_ylabel('y')
grafica.set_zlabel('z')
# grafica.legend()
grafica.view_init(45, 45)

# plt.figure()
# plt.title("Imagen 1 original")
# plt.imshow(dcm_eje_corto.get_imagen())

# plt.figure()
# plt.title("Imagen 2 original")
# plt.imshow(dcm_eje_largo.get_imagen())

plt.show()