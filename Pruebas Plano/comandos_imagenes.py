import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pydicom as dcm
from pydicom.data import get_testdata_file
from pydicom.pixel_data_handlers.util import apply_voi_lut
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

filtro_medio = np.ones((3,3))/9

 # Crear un filtro gaussiano
x = cv.getGaussianKernel(8,2)
filtro_gaussiano = x*x.T

 # Diferentes filtros de detección de bordes
 # scharr en dirección x
scharr = np.array([[-3, 0, 3],
                   [-10,0,10],
                   [-3, 0, 3]])

sharpen = np.array([[-1, -1, -1],
                    [-1,12,-1],
                    [-1, -1, -1]])

 # x dirección sobel
sobel_x= np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

 # sobel en y dirección
sobel_y= np.array([[-1,-2,-1],
                   [0, 0, 0],
                   [1, 2, 1]])

 # Transformada de Laplace
laplaciano=np.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]])

def fft2_centrada(imagen):
    # np.fft.fft2() nos proporciona conversión de frecuencia, será una matriz compleja. [F(u,v)]
    fourier = np.fft.fft2(imagen)

    #una vez que se obtiene el resultado, el componente de frecuencia cero (componente DC) se ubicará en la esquina superior izquierda.
    # con este comando ubicamos las bajas frecuencias en el centro¿
    centrada = np.fft.fftshift(fourier)

    return centrada

def inversa_fft2_centrada(centrada):
    fourier= np.fft.ifftshift(centrada)

    imagen = np.fft.ifft2(fourier)

    return imagen

def magnitud_espectro(fourier,c):
    #   D(u,v) = c . log[1+F(u,v)]        donde c es una constante de escalado.
    
    # De esta forma, la función logarítmica realiza una compresión del rango dinámico, facilitándose considerablemente la visualización
    # e interpretación del espectro de Fourier
    
    return c * np.log(1 + np.abs(fourier))


def agregar_ruido(noise_typ,image):
    if noise_typ == "gauss":

       row,col,ch= image.shape
       mean = 0
       var = 0.01
       sigma = var**0.5
       gauss = np.random.normal(mean,sigma,(row,col,ch))
       gauss = gauss.reshape(row,col,ch)
       noisy = image + gauss
       return noisy
    elif noise_typ == "s&p":
       row,col,ch = image.shape
       s_vs_p = 0.5
       amount = 0.004
       out = np.copy(image)

       # Salt mode
       num_salt = np.ceil(amount * image.size * s_vs_p)
       coords = [np.random.randint(0, i - 1, int(num_salt))
               for i in image.shape]
       out[coords] = 1

       # Pepper mode
       num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
       coords = [np.random.randint(0, i - 1, int(num_pepper))
               for i in image.shape]
       out[coords] = 0
       return out
    elif noise_typ == "poisson":
       vals = len(np.unique(image))
       vals = 2 ** np.ceil(np.log2(vals))
       noisy = np.random.poisson(image * vals) / float(vals)
       return noisy
    elif noise_typ =="speckle":
       row,col,ch = image.shape
       gauss = np.random.randn(row,col,ch)
       gauss = gauss.reshape(row,col,ch)        
       noisy = image + image * gauss
       return noisy


def estirarmiento_histograma(imagen, MIN, MAX, LSRD):
    #   Estiramiento de histograma (v-m)*LSRD/(M-m)donde  v  refiere  al  valor  del  píxel,  m  refiere  al  valor  mínimo  del  histograma,  LSRD  se 
    #   refiere  la  Límite  Superior  Resolución  Radiométrica  de  la  imagen,  y  M  refiere  al  valor máximo del histograma.
    return (((imagen-MIN)/(MAX-MIN))*LSRD)

def transformacion_exponencial(imagen, c, y):
    #La fórmula para aplicar la transformación exponencial en una imagen es
    #   S = c*r**y
    #   dónde
    #   R = valor de píxel de entrada
    #   C = constante de escala
    #   Y = ?
    #   S = valor de píxel de salida
    #
    #   255 / np.log(1 + np.amax(imagen)) 
    return np.array(c * imagen ** y, dtype=np.uint8)


def transformacion_logaritmica(imagen, c):
    #   La fórmula para aplicar la transformación logarítmica en una imagen es
    #   S = c * log(1 + r)
    #   dónde
    #   R = valor de píxel de entrada
    #   C = constante de escala
    #   S = valor de píxel de salida
    #
    #   255 / np.log(1 + np.amax(imagen))

    return np.array(c * (np.log(1 + imagen)), dtype=np.uint8)



def leer_imagen_dicom(path, voi_lut = True, fix_monochrome = True):
    dicom = dcm.dcmread(path)
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        imagen = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        imagen = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        imagen = np.amax(imagen) - imagen
        
    imagen = imagen - np.min(imagen)
    imagen = imagen / np.max(imagen)
    imagen = (imagen * 255).astype(np.uint8)
        
    return imagen

def cluster_kmeans(imagen, num_clusters=5, dtype='uint8'):
    """
    Imagen: Imagen a aplicar el cluster
    num_cluster: Número de clusters
    """
    switcher = {
        'uint8': 255,
        'uint16': 65535
    }
    print(np.size(imagen.shape))
    if (np.size(imagen.shape)>2):
        (filas, columnas, bandas) = imagen.shape
        img_r = imagen.reshape(-1, bandas)
    else:
        (filas, columnas) = imagen.shape
        img_r = imagen.reshape(-1,1)
    #Fit K-means on resized image. n_clusters is the desired number of colors 
    k_colors = KMeans(n_clusters=num_clusters).fit(img_r)
    #Assign colors to pixels based on their cluster center
    #Each row in k_colors.cluster_centers_ represents the RGB value of a cluster centroid
    #k_colors.labels_ contains the cluster that a pixel is assigned to
    #The following assigns every pixel the color of the centroid it is assigned to
    imagen_clusterizada = k_colors.cluster_centers_[k_colors.labels_]
    #Reshape the image back to 128x128x3 to save
    imagen_clusterizada = np.reshape(imagen_clusterizada, (imagen.shape))
    etiquetas = np.reshape(k_colors.labels_, (filas, columnas))
    return imagen_clusterizada, etiquetas

    
def cluster_dbscan(imagen, n_eps=0.30, n_min_samples=9):
    switcher = {
        'uint8': 255,
        'uint16': 65535
    }
    bandas = 0
    if (np.size(imagen.shape)>2):
        (filas, columnas, bandas) = imagen.shape
        # img_r = (imagen / switcher.get(dtype)).reshape(-1, bandas)
        img_r = imagen.reshape(-1, bandas)
    else:
        (filas, columnas) = imagen.shape
        # img_r = (imagen / switcher.get(dtype)).reshape(-1, bandas)
        # img_r = imagen.flatten()
        img_r = imagen.reshape(-1,1)
    
    X = img_r

    # define the model
    model = DBSCAN(eps=n_eps, min_samples=n_min_samples)

    # fit model and predict clusters
    yhat = model.fit_predict(X)

    # retrieve unique clusters
    clusters = np.unique(yhat)
    
    # # create scatter plot for samples from each cluster
    # for cluster in clusters:
    #     # get row indexes for samples with this cluster
    #     row_ix = where(yhat == cluster)
    #     # create scatter of these samples
    #     pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # # show the plot
    # pyplot.show()

    # imagen_clusterizada = np.reshape(clusters, (filas, columnas))
    if (np.size(imagen.shape)>2):
        imagen_clusterizada = yhat.reshape(filas, columnas, bandas)
    else:
        imagen_clusterizada = yhat.reshape(filas, columnas)
    etiquetas = np.reshape(yhat, (filas, columnas))
    return imagen_clusterizada, etiquetas
