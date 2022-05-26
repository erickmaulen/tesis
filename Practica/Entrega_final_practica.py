import gym
import numpy as np
import cv2
from PIL import Image
from skimage import data, filters
from gym.wrappers import Monitor

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
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)


env = gym.make('MsPacman-v0')
images = []
done = False
step = 0

env.reset()
while not done and step < 1000:
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
    video.write(images[i][0:height-40,0:width])

video.release()   


# Se abre el video
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

# Muestra el fondo del videojuego (estatico)
#cv2.imshow('Fondo', medianFrame)
#cv2.waitKey(20)
#cv2.destroyAllWindows()

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Convertir el fondo en escala de grises
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

blobs = []
ret = True

#Con esto se quiere comparar la extracion de fondo mediante este metodo y el manual
object_detector = cv2.createBackgroundSubtractorMOG2()

while(ret):

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
  contours, _ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
  for cnt in contours:
    #Con esta funcion se dibujan los conrtonos a los diferentes objetos detectados
    cv2.drawContours(dframe, [cnt],-1,(0,255,0),2)
    #Coordenadas de los obetos encontrados
    x, y, w, h = cv2.boundingRect(cnt)
    #Aca la idea es que se dibuje un rectangulo sobre el objeto 
    cv2.rectangle(frame,(x,y), (x +w,y +h),(0,255,0),3)
    #print(x,y)
  
   
  blob = mask
  blobs.append(blob)
  #Detectar los posibles Blobs (agrupacion de pixeles con similitudes)
  keypoints = detector.detect(mask)
  #Se detectan los blobs con circulos rojos 
  im_with_keypoints = cv2.drawKeypoints(mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  if ret == True:
    #num_labels,labels,stats,centroides = cv2.connectedComponentsWithStats(blob,4,cv2.CV_32S)
    #valor_max=(np.max(stats[:4[1:]]))/2
    #print(valor_max)
    gauss = cv2.GaussianBlur(mask, (5,5), 0)
    canny = cv2.Canny(gauss, 50, 150)
    cv2.imshow('canny',canny)
    (contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("He encontrado {} objetos".format(len(contornos)))
    cv2.drawContours(frame,contornos,-1,(0,0,255), 2)
    cv2.imshow("contornos", frame)
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






