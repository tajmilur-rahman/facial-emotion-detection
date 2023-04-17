
import pandas as pd 


# In[2]:


import cv2


# In[3]:


import numpy as np 
from sklearn.model_selection import train_test_split


# In[4]:


FILE_PATH = 'fer2013.csv'


# In[5]:


image_size = (48, 48)


# In[6]:


data = pd.read_csv(FILE_PATH)    


# In[7]:


data.head()


# In[8]:


pixels = data['pixels'].tolist()


# In[9]:


width, height = image_size


# In[10]:


# load images and emotions 
faces = []

for p in pixels:
    face = [int(pix) for pix in p.split(' ')]
    face = np.asarray(face).reshape(width, height)
    face = cv2.resize(face.astype('uint8'), image_size)
    faces.append(face.astype('float32'))


# In[11]:


len(faces)


# In[12]:


faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)


# In[ ]:





# In[13]:


emotions = pd.get_dummies(data['emotion']).values
emotions.shape


# In[14]:


# pre-process the images 
def preprocess(x, v2=True):  # v2 to keep the image btw. -1 and 1
    x = x.astype('float32')
    x = x/255.0
    if v2:
        x = (x - 0.5)*2.0
    return x


# In[15]:


faces = preprocess(faces)


# In[16]:


#print('showing some sample training images')
#for image in np.arange(0,10):
#    cv2.namedWindow('some sample training images', cv2.WINDOW_NORMAL)
#    cv2.imshow('some sample training images',faces[image])
#    cv2.waitKey(500)
#    cv2.destroyAllWindows()


# In[17]:


xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)


# 
# # CNN model : Mini Xception

# In[18]:


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


# In[23]:


#parameters 

batch_size = 32
epochs = 100
image_shape = (48, 48, 1)
verbose = True 
num_class = 7
patience = 50  # number of epochs with no improvement after which training will be stopped
base_path = 'models/'
l2_regularization = 0.01


# In[24]:


data_generator = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, rotation_range=10, 
                                    width_shift_range=0.1, height_shift_range=0.1, zoom_range=.1, horizontal_flip=True)


# In[25]:


regularization = l2(l2_regularization)


# In[26]:


# model
image_input = Input(image_shape)
x = Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularization, use_bias=False)(image_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# module 1
# residual module 
residual = Conv2D(filters=16, kernel_size=(1,1), strides=(2,2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = SeparableConv2D(filters=16, kernel_size=(3,3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(filters=16, kernel_size=(3,3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
x = layers.add([x,residual])

# module 2
# residual module 
residual = Conv2D(filters=32, kernel_size=(1,1), strides=(2,2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = SeparableConv2D(filters=32, kernel_size=(3,3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(filters=32, kernel_size=(3,3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
x = layers.add([x,residual])

# module 3
# residual module 
residual = Conv2D(filters=64, kernel_size=(1,1), strides=(2,2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = SeparableConv2D(filters=64, kernel_size=(3,3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(filters=64, kernel_size=(3,3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
x = layers.add([x,residual])

# module 4
# residual module 
residual = Conv2D(filters=128, kernel_size=(1,1), strides=(2,2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = SeparableConv2D(filters=128, kernel_size=(3,3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(filters=128, kernel_size=(3,3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
x = layers.add([x,residual])

x = Conv2D(filters=num_class, kernel_size=(3,3), padding='same')(x)
x = GlobalAveragePooling2D()(x)

output = Activation('softmax', name='predictions')(x)


# In[27]:


model = Model(image_input, output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[28]:


# callbacks 
log_file_path = base_path + '_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)

early_stop = EarlyStopping(monitor='val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=int(patience/4), verbose=verbose)

trained_models_path = base_path + '_mini_xception'
model_names = trained_models_path + '.{epoch:02d}_{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(filepath=model_names, monitor='val_loss', verbose=verbose, save_best_only=True)

callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]


# In[29]:


# Uncomment below line to train the model
'''model.fit_generator(data_generator.flow(xtrain, ytrain, batch_size), 
                    steps_per_epoch=len(xtrain)/batch_size, epochs=epochs, verbose=verbose, 
                    callbacks=callbacks, validation_data=(xtest, ytest))'''


# # Detecting emotions of a face in an image 

# In[30]:


from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import sys 


# In[31]:


detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_recognition_model_path = base_path + '_mini_xception.100_0.65.hdf5'
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




