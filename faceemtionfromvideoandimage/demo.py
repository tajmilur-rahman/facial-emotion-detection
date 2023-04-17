from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import sys 
import pandas as pd 


# In[2]:


import cv2


# In[3]:

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# In[19]:


from keras.preprocessing.image import ImageDataGenerator


# In[20]:


from keras import layers
from keras.layers import Activation, Convolution2D, Conv2D, Dropout, AveragePooling2D, BatchNormalization, GlobalAveragePooling2D, Flatten, Input, MaxPooling2D, SeparableConv2D


# In[21]:


from keras.models import Model


# In[22]:


from keras.regularizers import l2


import numpy as np 
from sklearn.model_selection import train_test_split


# In[4]:


# In[31]:


detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_recognition_model_path =  'models/_mini_xception.100_0.65.hdf5'
image_path = 'women smiling.jpg'


# In[32]:


face_detection = cv2.CascadeClassifier(detection_model_path)


# In[33]:


emotion_classifier = load_model(emotion_recognition_model_path)


# In[34]:


emotions = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']


# In[35]:


color_frame = cv2.imread(image_path)
gray_frame = cv2.imread(image_path, 0)


# In[36]:


cv2.imshow('Input test image', color_frame)
cv2.waitKey(1000)
cv2.destroyAllWindows()


# In[37]:


detected_faces = face_detection.detectMultiScale(color_frame, scaleFactor=1.1, minNeighbors=5, 
                                        minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
print('Number of faces detected : ', len(detected_faces))

if len(detected_faces)>0:
    
    detected_faces = sorted(detected_faces, reverse=True, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))[0] # if more than one faces
    (fx, fy, fw, fh) = detected_faces
    
    im = gray_frame[fy:fy+fh, fx:fx+fw]
    im = cv2.resize(im, (48,48))  # the model is trained on 48*48 pixel image 
    im = im.astype("float")/255.0
    im = img_to_array(im)
    im = np.expand_dims(im, axis=0)
    
    preds = emotion_classifier.predict(im)[0]
    emotion_probability = np.max(preds)
    label = emotions[preds.argmax()]
    
    cv2.putText(color_frame, label, (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.rectangle(color_frame, (fx, fy), (fx + fw, fy + fh),(0, 0, 255), 2)

cv2.imshow('Input test image', color_frame)
cv2.imwrite('output_'+image_path.split('/')[-1], color_frame)
cv2.waitKey(1000)
cv2.destroyAllWindows()


# In[ ]:





# # Detecting emotions of faces in a video

# In[45]:


cv2.namedWindow('emotion_recognition')
#camera = cv2.VideoCapture(0)  ## uncomment to use your laptop camera 
camera = cv2.VideoCapture('various_emotions.mp4')  # uncomment to read from a video file

sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.VideoWriter_fourcc(*'mpeg')

out = cv2.VideoWriter()
out.open('output_various_emotions.mp4',fourcc, 15, sz, True) # initialize the writer


# while True: # when reading from a video camera, use this while condition
while(camera.read()[0]):  # when reading from a video file, use this while condition
    color_frame = camera.read()[1]
    color_frame = imutils.resize(color_frame,width=min(720, color_frame.shape[1]))
    
    
    gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detection.detectMultiScale(gray_frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = color_frame.copy()    

    
    if len(detected_faces)>0:

        detected_faces = sorted(detected_faces, reverse=True, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))[0] # if more than one faces
        (fx, fy, fw, fh) = detected_faces

        im = gray_frame[fy:fy+fh, fx:fx+fw]
        im = cv2.resize(im, (48,48))  # the model is trained on 48*48 pixel image 
        im = im.astype("float")/255.0
        im = img_to_array(im)
        im = np.expand_dims(im, axis=0)

        preds = emotion_classifier.predict(im)[0]
        emotion_probability = np.max(preds)
        label = emotions[preds.argmax()]

        cv2.putText(color_frame, label, (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(color_frame, (fx, fy), (fx + fw, fy + fh),(0, 0, 255), 2)

    
    for (i, (emotion, prob)) in enumerate(zip(emotions, preds)):
        # construct the label text
        text = "{}: {:.2f}%".format(emotion, prob * 100)
        w = int(prob * 300)
        
        cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 50, 100), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frameClone, label, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 150, 100), 2)
        cv2.rectangle(frameClone, (fx, fy), (fx + fw, fy + fh), (100, 100, 100), 2)
    
    out.write(frameClone)
    out.write(canvas)
    
    cv2.imshow('emotion_recognition', frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
camera.release()
out.release()
cv2.destroyAllWindows()


# In[ ]:




