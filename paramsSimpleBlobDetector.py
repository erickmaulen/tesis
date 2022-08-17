import cv2

def SimpleBlobDetectorParams():
    # Aca se setean los parametros del detector de Blobs
    params = cv2.SimpleBlobDetector_Params()

    # Cambiar thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filtrar por Area
    params.filterByArea = True
    params.minArea = 0.5

    # Filtrar por circularidad
    params.filterByCircularity = True
    params.minCircularity = 0.7

    # Filtrar por convexidad
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Filtrar por relacion de inercia
    params.filterByInertia = True
    params.minInertiaRatio = 0.087

    return params