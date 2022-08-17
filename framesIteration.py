from addObjetos import AddObjects


def FramesIteration(centers,r,dframe,objetos,lastPosition,move):
#Ciclo de la magia, itera por cada objeto en el frame
    for i in range(0,len(centers)):   
            x = centers[i][0]
            y = centers[i][1]
            rectX = (x - r) 
            rectY = (y - r)
            if(rectX < 0):
                rectX = 0
            #Oldobject es un objeto , lo que hace es cortar segun el 
            #centro estimado del objeto y esa matriz asignarla a una variable
            objeto = dframe[rectY:(y+r),rectX:(x+r)]
            #Cuando captura un posible objeto pero un
            #un frame ya no esta, este if evita ese problema
            AddObjects(objeto,x,y,objetos,lastPosition,i,move)