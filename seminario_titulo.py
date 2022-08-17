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
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
from scipy.stats.stats import pearsonr
from addObjetos import AddObjects
from backgroundSubtraccion import BackgroundSubtraction
from calculateMediaFrame import CalculateMediaFrame
from correlacion import Correlacion
from diccionarios import Diccionarios
from drawContornos import DrawContornos
from framesIteration import FramesIteration
from move import Move
import paramsSimpleBlobDetector as p
from similitud import Similitud
import trainVideoRecord as t

# Setear parametros del detector de Blobs
params = p.SimpleBlobDetectorParams()
# Se crea el detector con los parametros
detector = cv2.SimpleBlobDetector_create(params)
accionesAgent,videoblob = t.trainVideoRecord()
grayMedianFrame,cap = CalculateMediaFrame()
blobs = []
ret = True
#Con esto se quiere comparar la extracion de fondo mediante este metodo y el manual
object_detector = cv2.createBackgroundSubtractorMOG2()
dictionaryAction,dicObjects,flag,objetos,lastPosition,move = Diccionarios(accionesAgent)

while(ret):
      im_with_keypoints,keypoints,mask,frame,dframe,flag = BackgroundSubtraction(cap,grayMedianFrame,object_detector,detector)
      if not flag:
            print("GAMEOVER")
            break
      DrawContornos(ret,mask,frame,keypoints,cap,videoblob,objetos,lastPosition,move,im_with_keypoints,dframe)
      cv2.waitKey(10)

#Encontrar la correlacion de las acciones de los diferentes objetos
#con las acciones cometidas por el agente
Correlacion(dictionaryAction,accionesAgent)
cap.release()
cv2.destroyAllWindows()
#Por ultimo se puede observar que hay dos videos, uno de lo que jugo la IA y otro de lo que se detecto a partir de los Blobs.
