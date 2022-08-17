
from math import dist
from move import Move


def Similitud(positionRelative,position,objetos,objeto,lastPosition,i,move):
    if dist(positionRelative, position) == 1: 
    #Preguntamos si existe una instancia de lista para agregar los objetos
        if isinstance(objetos[i], list):
                objetos[i].append(objeto)
                lastPosition[i].append(position)
                #Preguntamos si el arreglo move, el que guardara todos los cambios de
                #movimiento del objeto tiene una instancia de lista para agregar el movimiento.
                Move(positionRelative,position,move,i)
        else:
                objetos.append(list())
                objetos[i].append(objeto)
                lastPosition.append(list())
                lastPosition[i].append(position)
                if (positionRelative[0] > position[0]):
                    move.append(3)
                elif (positionRelative[0] < position[0]):
                    move.append(2)
                elif (positionRelative[1] > position[1]):
                    move.append(4)
                elif (positionRelative[1] < position[1]):
                    move.append(1)
            