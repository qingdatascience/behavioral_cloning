import os
import cv2
import json
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, Lambda, ELU, Dropout, MaxPooling2D, Activation
from keras.layers.convolutional import Cropping2D
from keras.optimizers import Adam
from keras.regularizers import l2, activity_l2
import tensorflow as tf
tf.python.control_flow_ops = tf

##Read the data
driving_log = pd.read_csv('./data/driving_log_train.csv')

driving_log_validation = pd.read_csv('./data/driving_log_validation.csv')

image_path = './data/IMG'
np.random.seed(2)

##Variables to be used in processing pipeline
driving_log = shuffle(driving_log)
driving_log_validation = shuffle(driving_log_validation)

  
ckp_threshold = 0.3
correction = 0.2

##Jittering augmentation
def augment_color_v(image):
    factor1= 0.75 + np.random.uniform(low=0.,high=0.75)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[:,:,2] = image[:,:,2] * factor1

    ##Clip the image so that no pixel has value greater than 255
    image[:,:,2] = np.clip(image[:,:,2], a_min=0, a_max=255)
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


## Define the steps here for how the pipeline is going to be exceuted
def preprocess_traindata(idx):
    steering_angle = driving_log.iloc[idx,3]
    j = np.random.randint(3)
    if j == 0:
        source_path = driving_log.iloc[idx,1].strip()
        tokens = source_path.split('/')
        image = cv2.imread(os.path.join(image_path, tokens[-1])) ##Left image
        steering_angle += correction
    elif j == 1:
        source_path = driving_log.iloc[idx,2].strip()
        tokens = source_path.split('/')    
        image = cv2.imread(os.path.join(image_path, tokens[-1])) ##Right image
        steering_angle -= correction
    else:
        source_path = driving_log.iloc[idx,0].strip()
        tokens = source_path.split('/')    
        image = cv2.imread(os.path.join(image_path, tokens[-1]))   ##Center image
 
    
    ##Apply flipping 
    f= np.random.randint(2)
    if f == 0:
        image = cv2.flip(image, 1)
        steering_angle *= -1

    ##Change from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # change brightness 
    k = np.random.randint(2)
    if k == 0:
        image = augment_color_v(image)

    return image, steering_angle    


##Preprocessing for validation data
def preprocess_validdata(idx):
    source_path = driving_log_validation.iloc[idx,0].strip()
    tokens = source_path.split('/')
    image = cv2.imread(os.path.join(image_path, tokens[-1]))

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    #plt.imshow(image)
    #plt.show()

    steering_angle = driving_log_validation.iloc[idx,3]
    return image, steering_angle  
    

## Define Generator
def data_generator_train(data, batch_size=32):
    batch_images = np.zeros((batch_size, 160, 320, 3), dtype=np.float32)
    batch_steering_angles = np.zeros((batch_size,),dtype=np.float32)
  
    while 1:
        for batch_idx in range(batch_size):
            ckp = 0

            while ckp == 0:
                idx = np.random.randint(len(data))
                img, st_angle = preprocess_traindata(idx)
                if abs(st_angle)<0.1:
                    ckp_ = np.random.uniform()
                    if ckp_ > ckp_threshold:
                        ckp = 1
                else:
                    ckp = 1

            batch_images[batch_idx] = img
            batch_steering_angles[batch_idx] = st_angle
            
                  
        yield batch_images, batch_steering_angles

##Validation image generator
def data_generator_validation(data, batch_size=32):
    batch_images = np.zeros((batch_size, 160, 320, 3), dtype=np.float32)
    batch_steering_angles = np.zeros((batch_size,),dtype=np.float32)    
    while 1:
        for i in range(batch_size):
            idx = np.random.randint(len(data))
            img, st_angle = preprocess_validdata(idx)
            batch_images[i] = img
            batch_steering_angles[i] = st_angle
                       
        yield batch_images, batch_steering_angles                   

#preprocess_traindata(1255)
#preprocess_validdata(225)
#exit()      


##Generators
train_data_gen = data_generator_train(driving_log)
val_data_gen = data_generator_validation(driving_log_validation)


##Build Nvidia's architecture with modifications

model = Sequential()

model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid"))
model.add(Activation('relu'))


model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid"))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid"))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode="valid"))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode="valid"))
model.add(Activation('relu'))


model.add(Flatten())


model.add(Dense(100, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(50, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10, init='glorot_uniform'))
model.add(Activation('relu'))
model.add(Dense(1))
model.compile(loss="mse", optimizer='adam')

#model.summary()


##Train the model
model.fit_generator(train_data_gen, samples_per_epoch=23296,validation_data= val_data_gen, verbose=1, 
                                             nb_val_samples=2000, nb_epoch=10)

##Save the model weights and json file
#model_json = model.to_json()

#with open(model_name+'.json', "w") as json_file:
 #   json_file.write(model_json)
#model.save_weights('model.h5')
model.save('Mymodel.h5')