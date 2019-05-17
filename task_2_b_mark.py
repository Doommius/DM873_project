from keras import preprocessing as pr
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from pandas import DataFrame
from keras.models import load_model
import pandas as pd

##printing option for PD
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

'''
BEGIN CONFIG!
'''
samples = ["Nymphalidae", "Lycaenidae"]
IMG_SIZE = 224
EPOCHS = 3
BATCH_SIZE = 32
# 1/scale
scale = 32
learning_rate = 1e-3

'''
END CONFIG!
'''
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



num_classes = len(samples)
df1 = df.loc[df['family'].isin(["Nymphalidae"])].sample(n=250)
df2 = df.loc[df['family'].isin(["Lycaenidae"])].sample(n=250)

##from data frame to data generator
train_dataframe = df1.append(df2)
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
                                                    y_col="family", class_mode="categorical",
                                                    target_size=(IMG_SIZE, IMG_SIZE),
                                                    batch_size=BATCH_SIZE)

validation_generator = test_datagen.flow_from_dataframe(validation_dataframe, directory="dataset/base_set",
                                                        x_col='filename', y_col="family", class_mode="categorical",
                                                        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)
# input layer

model = load_model('task_2_a_mark_checkpoint.h5')


es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=3)

mc = ModelCheckpoint(
    "task_2_b_mark_checkpoint.h5",
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_best_only=True)

##how to freeze layers.




for layer in model.layers[:12]:
    layer.trainable=False
for layer in model.layers[12:]:
    layer.trainable=True

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())

plot_model(model, to_file='network.png')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=2)

plot_training(history)

model.save_weights("task_2_b_mark.h5")
