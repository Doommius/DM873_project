from keras import preprocessing as pr
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from pandas import DataFrame


train_datagen = pr.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = pr.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'task1/a/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

validation_generator = train_datagen.flow_from_directory(
    'task1/a/validate',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')



model = Sequential()
model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(224, 224, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(224, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    samples_per_epoch=60,
    epochs=10,
    validation_data=validation_generator,
    nb_val_samples=400)

# model.save_weights("task1.h5")