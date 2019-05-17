from keras import preprocessing as pr
import pandas as pd
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from pandas import DataFrame
import os
import tensorflow as tf


import matplotlib.pyplot as plt
def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    # plt.figure()
    # plt.plot(epochs, loss, 'r.')
    # plt.plot(epochs, val_loss, 'r-')
    # plt.title('Training and validation loss')
    plt.show()

    plt.savefig('acc_vs_epochs.png')


IMG_SIZE = 224
BATCH_SIZE = 32

train_datagen = pr.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale = 1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = pr.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'task1/a/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary')

validation_generator = train_datagen.flow_from_directory(
    'task1/a/validate',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary')


model = Sequential()

model.add(Conv2D(
        filters=64,
        kernel_size=(3,3),
        strides=1,
        padding='same',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        data_format='channels_last',
        activation='relu'))

model.add(MaxPooling2D(
        pool_size=(2,2),
        strides=2))
    
model.add(Dropout(
        rate=0.2))

model.add(Conv2D(filters=64,
               kernel_size=(2,2),
               strides=(1,1),
               padding='valid'))

model.add(Activation('relu'))
model.add(MaxPooling2D(
                    pool_size=(2,2),
                    strides=2))

model.add(Flatten())        
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dropout(
        rate=0.2))

model.add(Dense(1))
model.add(Activation('sigmoid'))

es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=3)

mc = ModelCheckpoint(
        "task_1_morten.h5",
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


history = model.fit_generator(
        train_generator,
        steps_per_epoch=60,
        epochs=15,
        validation_data=validation_generator,
        nb_val_samples=10,
        callbacks=[es, mc])

plot_training(history)