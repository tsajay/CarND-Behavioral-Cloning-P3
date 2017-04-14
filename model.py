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
from random import randint

parser = argparse.ArgumentParser(description='Train a model in Keras and TensorFlow.')
parser.add_argument('-c', '--csv_file', required=True, help='Path to the csv file with training data and path to images.')
parser.add_argument('-i', '--img_dir', required=True, help='Path to the images directory.')
parser.add_argument('-m', '--model_load', help='A model to pre-load for training.')
parser.add_argument('-e', '--epochs', default=1, help='Number of epochs for training.')
parser.add_argument('-s', '--save_model', default="model.h5", help='Name for the final saved model.')
parser.add_argument('-p', '--checkpoint', action='store_true', help='Checkpoint intermediate epochs.')
parser.add_argument('-n', '--checkpoint_name', default='cp', help='Checkpoint name (suffix-only).')
parser.add_argument('-o', '--only_center', action='store_true', help='Only center images for training')
parser.add_argument('-l', '--learning_rate', default=0.001, help='Optional learning rate for Adam optimizer (default = 0.001)')
parser.add_argument('-b', '--batch_size', default=32, help='Optional learning rate for Adam optimizer (default = 0.001)')

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
camera_tilts = [0.02, 0.2, -0.2]


from sklearn.utils import shuffle
shuffle(samples)

from sklearn.model_selection import train_test_split
from scipy.ndimage.interpolation import shift 
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


'''
image is a 3-D vector. width X height X num_channels

Returns a 3-D vector of the same size. Images are shifted randomly by 5 pixels either up/down/left/right
'''
def getAugmentedImageWithShifts(image):
    
    shift_vectors = [[0, -5, 0], # Up 
                     [0,  5, 0], # Down
                     [-5, 0, 0], # Left
                     [5,  0, 0]] # Right
    
    direction = randint(0,3)
    image = shift(image, shift_vectors[direction])
    
    return image


def generator(samples,batch_size=32, validation_gen=False,  only_center=False):

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
                camera_range = 3
                if only_center:
                    camera_range = 1
                for c_pos in range(3):
                    
                    source_path = next_sample[c_pos]
                    filename = source_path.split('/')[-1]
                    img_path = args.img_dir + '/' + filename
                    if (not os.path.isfile(img_path)):
                        print ("Unable to find the training image %s in the img-dir %s. Please check your args" %(filename, args.img_dir))
                        exit(1)
                    image = cv2.imread(img_path)
                    if (c_pos == 0):
                        image = getAugmentedImageWithShifts(image)

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
                if (not validation_gen or only_center):
                    a_measurements = [(measurement * - 1.0 + (np.random.random()/20.0 - 0.05)) for measurement in batch_measurements]
                    # a_measurements[0] += 0.025
                else:
                    # No randomness for validation.
                    a_measurements = [(measurement * - 1.0) for measurement in batch_measurements]
                ret_batch_images = np.append(batch_images, a_images, axis = 0)
                ret_batch_measurements = np.append(batch_measurements, a_measurements, axis = 0)
                current_batch_size = current_batch_size + rem
            offset += 1

            # trim image to only see section with road
            X_train_batch = np.array(ret_batch_images)
            y_train_batch = np.array(ret_batch_measurements)
            yield shuffle(X_train_batch, y_train_batch)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=args.batch_size, validation_gen=False, only_center=args.only_center)
validation_generator = generator(validation_samples, batch_size=args.batch_size, validation_gen=True, only_center=args.only_center)



'''
print ("Images-shape: %s" %(len(images)))
print ("a_images_shape: %s" %(len(a_images)))
print ("X_train-shape :" + str(X_train.shape))
print ("y_train_shape :" + str(y_train.shape))

'''


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers  import Adam


def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    #model.add(BatchNormalization())

    # Input-shape = 160 x 320 x 3. Output-shape = 65 x 320 x 3
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    
    # Input-shape = 65 x 320 x 3. Output-shape = 61 x 316 x 4
    model.add(Convolution2D(4,5,5,subsample=(1,1), activation='relu'))
    # Input-shape = 61 x 316 x 4. Output-shape = 30 x 158 x 4
    model.add(MaxPooling2D(strides=(2,2)))
    
    # Input-shape = 30 x 158 x 4. Output-shape = 26 x 154 x 6
    model.add(Convolution2D(6,5,5, subsample=(1,1), activation='relu'))
    # Input-shape = 26 x 154 x 6. Output-shape = 13 x 72 x 6
    model.add(MaxPooling2D(strides=(2,2)))

    # Input-shape = 13 x 72 x 6. Output-shape = 9 x 36 x 8
    model.add(Convolution2D(8,5,5, subsample=(1,1), activation='relu'))
    # Input-shape = 33 x 73 x 8. Output-shape = 16 x 36 x 8
    #model.add(MaxPooling2D(strides=(2,2)))
    
    # Input-shape = 9 x 36 x 8 . Output-shape = 7 x 33 x 10
    model.add(Convolution2D(10,3,3, subsample=(1,1), activation='relu'))
    # Input-shape = 7 x 33 x 10 . Output-shape = 3 x 16 x 10
    model.add(MaxPooling2D(strides=(2,2)))

    # Input-shape = 3 x 16 x 10. Output-shape = 1 x 14 x 10
    model.add(Convolution2D(64,3,3, subsample=(1,1), activation='relu'))
    
    # Input-shape = 1 x 14 x 10, Output-shape = 140
    model.add(Flatten())

    # Input-shape = 140, Output-shape = 128
    model.add(Dense(128))
    model.add(Dropout(0.15))

    # Input-shape = 128, Output-shape = 64
    model.add(Dense(64))
    model.add(Dropout(0.15))

    # Input-shape = 64, Output-shape = 32
    model.add(Dense(32))
    model.add(Dropout(0.15))

    # Input-shape = 32, Output-shape = 10
    model.add(Dense(10))

    # Final steering angle
    model.add(Dense(1))

    adam_optimizer = Adam(lr=args.learning_rate)

    model.compile(loss='mse', optimizer=adam_optimizer)
    
    model.summary()
    return model

# Allow for training special cases from an already trained network.
# Loads an existing model.
if (not args.model_load is None):
    model = load_model(args.model_load)
else:
    model = build_model()

from keras.utils import visualize_util

visualize_util.plot(model, to_file='model.png')
exit()

# Each row of 3 images generates 8 training/validation images. 
aug_factor = 8

# For debug - discard side camera images.
if args.only_center:
    aug_factor = 4

# Allow checkpointing intermediate epochs.
if (args.checkpoint):    
    checkpointer = ModelCheckpoint(filepath= "%s.{epoch:02d}-{val_loss:.4f}.hdf5" %args.checkpoint_name)
    model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples) * aug_factor, validation_data=validation_generator, \
            nb_val_samples=len(validation_samples) * aug_factor, nb_epoch=int(args.epochs), verbose=1
            , callbacks=[checkpointer])
else:
    model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples) * aug_factor, validation_data=validation_generator, \
            nb_val_samples=len(validation_samples) * aug_factor, nb_epoch=int(args.epochs), verbose=1)

# Always save the final model.
model.save(args.save_model)
exit()
