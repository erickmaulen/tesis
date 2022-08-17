import cv2
import numpy as np
from similitud import Similitud


def AddObjects(objeto,x,y,objetos,lastPosition,i,move):
    if objeto.size == 0:
        return False
    else:
        cv2.imwrite('img/objetos'+str(x)+'_'+str(y)+'.png', objeto) # Guardar la imagen de los objetos 
    #Posicion de un objeto
        position = x,y
        #En una primera instancia el largo del array es 0
        #Entonces agrega de inmediato el obejtos y su posicion
        if len(objetos) == 0:
                objetos.append(list())
                objetos[i].append(objeto)
                lastPosition.append(list())
                lastPosition[i].append(position)
        else:
                #En caso contrario primero se pregunta si el largo del array de objetos 
                #Es igual a i, esto debido a que si se cumple, es porque se encontro un objeto nuevo
                #Por lo tanto se agrega el objeto y su posicion
                if len(objetos) == i :
                    objetos.append(list())
                    objetos[i].append(objeto)
                    lastPosition.append(list())
                    lastPosition[i].append(position)
                    move.append(list())
                else:
                    #En caso de que difiera el largo, se debe recorrer el array con los diferentes objetos
                    #Con el fin de encontrar si el objeto actual ya se encuentra en el array
                    for j in range(len(objetos)):
                            positionRelative = lastPosition[i][-1] 
                            objetAArray = np.array(objetos[i][0])
                            objetBArray = np.array(objeto)
                            #Se calcula la distancia entre el nuevo objeto y cada ultima posicion 
                            #de los diferentes obejtos, en caso de ser 1 es el mismo objeto, solo
                            #que se movio un pixel
                            Similitud(positionRelative,position,objetos,objeto,lastPosition,i,move)
                    objetos.append(list())
                    objetos[i].append(objeto)
                    lastPosition.append(list())
                    lastPosition[i].append(position)