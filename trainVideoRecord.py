import gym
import cv2
import argparse
import imutils
from PIL import Image
from skimage import data, filters
from gym.wrappers import Monitor

def trainVideoRecord():   
    env = gym.make('ALE/MsPacman-v5',full_action_space=False)
    images = []
    done = False
    step = 0
    posiblesActions = [1,2,3,4]
    env.reset()
    accionesAgent = list()
    while not done and step < 10:
        action = env.action_space.sample()
        if action in posiblesActions:
                if len(accionesAgent) == 0:
                    accionesAgent.append(action)
                else :
                    if action == accionesAgent[-1]:
                            continue
                    else:
                            accionesAgent.append(action)
                    
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

    return accionesAgent, videoblob