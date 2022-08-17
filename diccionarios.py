


def Diccionarios(accionesAgent):
    flag = True
    posiblesObjects = []
    lastPosition = list()
    objetos = list()
    actions = list()
    move = list()
    dicObjects = {
        "Objetos" : objetos,
        "Posiciones" :lastPosition
    }
    dictionaryAction = {
        "Accion" : accionesAgent,
        "Move" : move
    }

    return dictionaryAction,dicObjects,flag,objetos,lastPosition,move