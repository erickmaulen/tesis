import gym
import numpy as np
import cv2
from PIL import Image
from skimage import data, filters
from gym.wrappers import Monitor

from skimage import measure
import argparse
import imutils
import cv2
from math import dist




def mse(imageA, imageB):
    	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float( np.mat(imageA.shape[0]) * np.mat(imageA.shape[1])) 
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

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

# Se crea el detector con los parametros
detector = cv2.SimpleBlobDetector_create(params)

env = gym.make('MsPacman-v0')
images = []
done = False
step = 0

env.reset()
while not done and step < 10:
    action = env.action_space.sample()
    state_next, reward, done, info = env.step(action)
    image = state_next
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    images.append(image)

#Aca se obtiene el alto y ancho debido a que se debera recortar la imagen final
height, width = state_next.shape[0:2]
video = cv2.VideoWriter('Ms-Pacman-v0.wmv',cv2.VideoWriter_fourcc(*'mp4v'),25,(width,height-40))

#La idea de este video es que muestre como hace tracking a los diferentes objetos que tiene el juego Pacman
videoblob = cv2.VideoWriter('Ms-Pacman-v0Blob.wmv',cv2.VideoWriter_fourcc(*'mp4v'),25,(width,height-40))

for i in range(len(images)):
    #print(len(images))
    video.write(images[i][0:height-40,0:width])

video.release()   


# Se abre el video
cap = cv2.VideoCapture("Ms-Pacman-v0.wmv")

# Seleccion random de 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=12)

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

blobs = []
ret = True

#Con esto se quiere comparar la extracion de fondo mediante este metodo y el manual
object_detector = cv2.createBackgroundSubtractorMOG2()
k = 0
flag = True
posiblesObjects = []

while(ret):
  #Aca solo ocupo el k para ver en que frame me encuentro
  k+=1
  # print(k)

  # Leer el frame
  ret, frame = cap.read()
  if not ret:
    print("GAMEOVER")
    break
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
  blob = mask
  blobs.append(blob)
  #Detectar los posibles Blobs (agrupacion de pixeles con similitudes)
  keypoints = detector.detect(mask)
  #Se detectan los blobs con circulos rojos 
  im_with_keypoints = cv2.drawKeypoints(mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  #Si hay frame
  if ret == True:
    gauss = cv2.GaussianBlur(mask, (5,5), 0)
    canny = cv2.Canny(gauss, 50, 150)
    (contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame,contornos,-1,(0,0,255), 2)   
    for key in keypoints:
      x = key.pt[0]
      y = key.pt[1]
 
    centers = []
    lastPosition = []
    dicObjects = {
      "Objetos" : posiblesObjects,
      "Posiciones" :lastPosition
    }
    r = 5
    #Ciclo para dibujar los circulos en cada objeto
    for i in range(0,len(contornos)):
      (rx, ry), radio = cv2.minEnclosingCircle(contornos[i])
      center = (int(rx),int(ry))
      centers.append(center)
      radius = int(r)
      dframe = cv2.circle(dframe,center,9,(255,0,0),0)
      cv2.imshow('dfa',dframe)
      
    #Ciclo de la magia
    for i in range(0,len(centers)):
      print("iteracion ",i)
      print("cantidad de objetos en este frame",len(centers))
      
      #print(len(contornos))    
      x = centers[i][0]
      y = centers[i][1]
      rectX = (x - r) 
      rectY = (y - r)
      if(rectX < 0):
          rectX = 0
      #Posicion de un objeto
      position = x,y
      #Oldobject es un objeto , lo que hace es cortar segun el 
      #centro estimado del objeto y esa matriz asignarla a una variable
      OldObject = dframe[rectY:(y+r),rectX:(x+r)]
      posiblesObjects.append(OldObject)
      #Cuando captura un posible objeto pero un
      #un frame ya no esta, este if evita ese problema
      if OldObject.size == 0:
        continue
      else:
      #Aca es donde se deberia hacer el diccionario de objetos
      #Esta flag era para que solo entrase la primera iteracion xd
        if flag:
          np.append(posiblesObjects[i],OldObject)
          #posiblesObjects[i].append(OldObject)
          lastPosition.append(position)
          
      
    flag = False    
    videoblob.write(im_with_keypoints)
    
    #Aca se muestra frame a frame como va jugando la IA y detectando los diferentes objetos
    cv2.imshow('dframe', im_with_keypoints)
    
  else:
      cap.release()
      videoblob.release()
      break
  
  # Mostrar frame que ocupo el metodo de cv2 para la deteccion de objetos 
  #cv2.imshow('frame', frame)
  cv2.waitKey(20)

video.release()  
cap.release()
cv2.destroyAllWindows()


#Por ultimo se puede observar que hay dos videos, uno de lo que jugo la IA y otro de lo que se detecto a partir de los Blobs.