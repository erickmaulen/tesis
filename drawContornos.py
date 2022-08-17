
import cv2

from framesIteration import FramesIteration


def DrawContornos(ret,mask,frame,keypoints,cap,videoblob,objetos,lastPosition,move,im_with_keypoints,dframe):
    if ret == True:
            gauss = cv2.GaussianBlur(mask, (5,5), 0)
            canny = cv2.Canny(gauss, 50, 150)
            (contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame,contornos,-1,(0,0,255), 2)   
            for key in keypoints:
                x = key.pt[0]
                y = key.pt[1]
            centers = []
            r = 5
            #Ciclo para dibujar los circulos en cada objeto
            for i in range(0,len(contornos)):
                (rx, ry), radio = cv2.minEnclosingCircle(contornos[i])
                center = (int(rx),int(ry))
                centers.append(center)
                radius = int(r)
                dframe = cv2.circle(dframe,center,9,(255,0,0),0)
                cv2.imshow('dfa',dframe)

            # centers,dframe,flag = DrawContornos(ret,mask,frame,keypoints,cap,videoblob)
            #Ciclo de la magia, itera por cada objeto en el frame

            FramesIteration(centers,r,dframe,objetos,lastPosition,move)
            videoblob.write(im_with_keypoints)
            #Aca se muestra frame a frame como va jugando la IA y detectando los diferentes objetos
            cv2.imshow('dframe', im_with_keypoints)
    else:
        cap.release()
        videoblob.release()
