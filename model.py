import argparse
import base64
from datetime import datetime
import os
import shutil
import copy

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import csv
import cv2
from keras.models import load_model
import h5py
from keras import __version__ as keras_version

parser = argparse.ArgumentParser(description='Train a model in Keras.')
parser.add_argument('-c', '--csv_file', required=True, help='Path to the csv file with training data and path to images.')
parser.add_argument('-i', '--img_dir', required=True, help='Path to the images directory.')

args = parser.parse_args()

if (not os.path.isfile(args.csv_file)):
    print ("Unable to find %s, Require a valid CSV file with training data as input." %args.csv_file)
    exit(1)

if (not os.path.isdir(args.img_dir)):
    print ("Require a valid directory with images for training the input.")
    exit(1)


samples = []
with  open(args.csv_file, "r") as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        samples.append(line)

images = []
measurements = []    
camera_tilts = [0.0, 0.2, -0.2]


from sklearn.utils import shuffle
shuffle(samples)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        offset = 0
        while offset < len(samples):
            next_sample = samples[offset]
            batch_images = []
            batch_measurements = []
            current_batch_size = 0
            while current_batch_size < batch_size:
                rem = batch_size - current_batch_size
                if (rem >= 8):
                    rem = 8
                # Add the center, left and right images.
                for c_pos in range(3):
                    
                    source_path = next_sample[c_pos]
                    filename = source_path.split('/')[-1]
                    img_path = args.img_dir + '/' + filename
                    if (not os.path.isfile(img_path)):
                        print ("Unable to find the training image %s in the img-dir %s. Please check your args" %(filename, args.img_dir))
                        exit(1)
                    image = cv2.imread(img_path)

                    batch_images.append(image)
                    # Using only the center image 
                    measurement = float(next_sample[c_pos + 3]) + camera_tilts[c_pos]
                    batch_measurements.append(measurement)
                    if (c_pos == 0):
                        batch_images.append(image)
                        measurement = float(next_sample[3]) 
                        batch_measurements.append(measurement)
                # Add the mirror images for a given image.
                a_images = [cv2.flip(im, 1) for im in batch_images]
                #a_measurements = [(measurement * - 1.0 + (np.random.random()/10.0 - 0.1)) for measurement in batch_measurements]
                a_measurements = [(measurement * - 1.0) for measurement in batch_measurements]
                # a_measurements[0] += 0.05
                ret_batch_images = np.append(batch_images, a_images, axis = 0)
                ret_batch_measurements = np.append(batch_measurements, a_measurements, axis = 0)
                current_batch_size = current_batch_size + rem
            offset += 1

            # trim image to only see section with road
            X_train_batch = np.array(ret_batch_images)
            y_train_batch = np.array(ret_batch_measurements)
            yield shuffle(X_train_batch, y_train_batch)

def val_generator(samples, batch_size=32):

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        offset = 0
        while offset < len(samples):
            next_sample = samples[offset]
            batch_images = []
            batch_measurements = []
            current_batch_size = 0
            while current_batch_size < batch_size:
                rem = batch_size - current_batch_size
                if (rem >= 8):
                    rem = 8
                # Add the center, left and right images.
                for c_pos in range(3):
                    
                    source_path = next_sample[c_pos]
                    filename = source_path.split('/')[-1]
                    img_path = args.img_dir + '/' + filename
                    if (not os.path.isfile(img_path)):
                        print ("Unable to find the training image %s in the img-dir %s. Please check your args" %(filename, args.img_dir))
                        exit(1)
                    image = cv2.imread(img_path)

                    batch_images.append(image)
                    # Using only the center image 
                    measurement = float(next_sample[c_pos + 3]) + camera_tilts[c_pos]
                    batch_measurements.append(measurement)
                    if (c_pos == 0):
                        batch_images.append(image)
                        measurement = float(next_sample[3]) 
                        batch_measurements.append(measurement)
                # Add the mirror images for a given image.
                a_images = [cv2.flip(im, 1) for im in batch_images]
                a_measurements = [(measurement * - 1.0) for measurement in batch_measurements]
                a_measurements[0] = a_measurements[0] + 0.8
                ret_batch_images = np.append(batch_images, a_images, axis = 0)
                ret_batch_measurements = np.append(batch_measurements, a_measurements, axis = 0)
                current_batch_size = current_batch_size + rem
            offset += 1

            # trim image to only see section with road
            X_train_batch = np.array(ret_batch_images)
            y_train_batch = np.array(ret_batch_measurements)
            yield shuffle(X_train_batch, y_train_batch)            

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)



'''
print ("Images-shape: %s" %(len(images)))
print ("a_images_shape: %s" %(len(a_images)))
print ("X_train-shape :" + str(X_train.shape))
print ("y_train_shape :" + str(y_train.shape))

'''


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
# model.add(Convolution2D(3,5,5, subsample=(2,2), activation='relu'))
# model.add(Flatten(input_shape=(160, 320, 3)))
model.add(AveragePooling2D(strides=(2,2)))
model.add(Convolution2D(4,3,3,activation='relu'))
model.add(MaxPooling2D(strides=(2,2)))
#model.add(Dropout(0.25))

model.add(Convolution2D(8,3,3,activation='relu'))
model.add(Dropout(0.05))

model.add(Convolution2D(4 ,3,3,activation='relu'))
model.add(Dropout(0.05))

model.add(Convolution2D(4 ,3,3,activation='relu'))
model.add(Dropout(0.05))

model.add(Convolution2D(4 ,3,3,activation='relu'))
model.add(Dropout(0.05))


model.add(Convolution2D(4,3,3,activation='relu'))
model.add(Dropout(0.25))

model.add(Convolution2D(4,3,3,activation='relu'))
model.add(Dropout(0.05))

#model.add(MaxPooling2D())
#model.add(Convolution2D(96,5,5,activation='relu'))
#model.add(Dropout(0.05))
#model.add(Convolution2D(18,5,5,activation='relu'))
model.add(Flatten())
# model.add(Dense(400))
model.add(Dense(40))
model.add(Dropout(0.05))
#model.add(Dense(160))
#model.add(Dropout(0.05))
model.add(Dense(10))
#model.add(Dropout(0.05))
#model.add(Dense(20))
#model.add(Dropout(0.05))
#model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7, batch_size=16)
# model.fit_generator(train_generator, samples_per_epoch=len(X_train)
#            len(train_samples), validation_data=validation_generator, /
#            nb_val_samples=len(validation_samples), nb_epoch=3)
# fit_generator(generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, \ 
# validation_data=None, validation_steps=None, )

model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples) * 8, validation_data=validation_generator, \
            nb_val_samples=len(validation_samples) * 8, nb_epoch=2, verbose=1
            )



model.save('model.h5')
exit()
