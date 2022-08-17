import cv2
import numpy as np

def CalculateMediaFrame():
    cap = cv2.VideoCapture("Ms-Pacman-v0.wmv")
    # Seleccion random de 25 frames
    frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

    frames = []
    for fid in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        frames.append(frame)


    # Calcular la mediana a lo largo del eje temporal
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Convertir el fondo en escala de grises
    grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

    return grayMedianFrame,cap
