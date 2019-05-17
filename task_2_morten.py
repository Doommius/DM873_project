from keras import preprocessing as pr
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt

##printing option for PD
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


##

# Plot the training and validation loss + accuracy
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


# load data from file to dataframe and change family to string and not int.
df = pd.read_csv("dataset/butterflies.txt", sep='\t')
df = df.drop(columns=['species', 'genus', 'subfamily'])

df.loc[df['family'].isin([1]), 'family'] = "Papilionidae"
df.loc[df['family'].isin([2]), 'family'] = "Pieridae"
df.loc[df['family'].isin([3]), 'family'] = "Nymphalidae"
df.loc[df['family'].isin([4]), 'family'] = "Lycaenidae"
df.loc[df['family'].isin([5]), 'family'] = "Hesperiidae"

samples = ["Nymphalidae", "Lycaenidae"]
IMG_SIZE = 224
BATCH_SIZE = 32
# 1/scale
scale = 32

num_classes = len(samples)
df = df.loc[df['family'].isin(samples)]
print(df)

##from data frame to data generator
train_dataframe = pd.DataFrame.sample(df, frac=0.7)
validation_dataframe = pd.DataFrame.drop(df, train_dataframe.index)

train_datagen = pr.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / scale,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = pr.image.ImageDataGenerator(rescale=1. / scale)

train_generator = train_datagen.flow_from_dataframe(train_dataframe, directory="dataset/base_set", x_col='filename',
                                                    y_col="family", class_mode="binary",
                                                    target_size=(IMG_SIZE, IMG_SIZE),
                                                    batch_size=BATCH_SIZE)

validation_generator = test_datagen.flow_from_dataframe(validation_dataframe, directory="dataset/base_set",
                                                        x_col='filename', y_col="family", class_mode="binary",
                                                        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)
# input layer
from keras.models import load_model
model = load_model('task_1_morten.h5')

# print(model.summary())

# plot_model(model, to_file='network.png')

# 
for layer in model.layers[:8]:
    print(layer.name, "should not train")
    layer.trainable=False





es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=3)

mc = ModelCheckpoint(
    "task_2_morten.h5",
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

# model.save_weights("task_2_b_mark.h5")
