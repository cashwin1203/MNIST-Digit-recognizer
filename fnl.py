import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

drawing = False 
mode = True 
ix,iy = -1,-1


def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.circle(img,(x,y),15,(255,0,0),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.circle(img,(x,y),15,(255,0,0),-1)


img = np.zeros([280,280],dtype ='uint8')
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
    elif k == ord('q'):
        break
    elif k == ord('c'):
        img[0:400,0:400]=0
    elif k == ord('s'):
        out = img[0:400,0:400]
        cv2.imwrite('Number.jpg',out)

cv2.destroyAllWindows()

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

m_new = tf.keras.models.load_model('mdl2.hdf5')


image_test = cv2.imread('Number.jpg',1)

image_test1 = cv2.resize(image_test,(28,28))

image_test_resize = image_test1.reshape(-1,28,28)

pred =np.argmax(m_new.predict(image_test_resize), axis=-1)
print(pred[0])










