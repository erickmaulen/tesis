


def Move(positionRelative,position,move,i):
    if isinstance(move[i], list):
        if (positionRelative[0] > position[0]):
            move[i].append(3)
        elif (positionRelative[0] < position[0]):
            move[i].append(2)
        elif (positionRelative[1] > position[1]):
            move[i].append(4)
        elif (positionRelative[1] < position[1]):
            move[i].append(1)
            
    else:
        move.append(list())
        if (positionRelative[0] > position[0]):
                move[i].append(3)
        elif (positionRelative[0] < position[0]):
                move[i].append(2)
        elif (positionRelative[1] > position[1]):
                move[i].append(4)
        elif (positionRelative[1] < position[1]):
                move[i].append(1)
    
