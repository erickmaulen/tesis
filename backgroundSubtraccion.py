import cv2
import numpy as np


def BackgroundSubtraction(cap,grayMedianFrame,object_detector,detector):
    ret, frame = cap.read()
    if not ret:
        print("GAMEOVER")
        return False
    # Convertir el actual frame en escala de grises
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Se calcula la diferencia entre el frame actual y la media(fondo)
    dframe = cv2.absdiff(frame, grayMedianFrame)
    # Treshold a binario
    th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
    #Deteccion de objetos
    mask = object_detector.apply(dframe)
    _, mask = cv2.threshold(mask,254,255, cv2.THRESH_BINARY)
    #contours, _ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # blob = mask
    # blobs.append(blob)
    #Detectar los posibles Blobs (agrupacion de pixeles con similitudes)
    keypoints = detector.detect(mask)
    #Se detectan los blobs con circulos rojos 
    im_with_keypoints = cv2.drawKeypoints(mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #Si hay frame
    return  im_with_keypoints,keypoints,mask,frame,dframe,True