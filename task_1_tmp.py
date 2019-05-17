from keras import preprocessing as pr
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import *
from keras import optimizers
from pandas import DataFrame
import pandas as pd

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

samples = ["Pieridae", "Papilionidae"]
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
                                                    y_col="family", class_mode="categorical",
                                                    target_size=(IMG_SIZE, IMG_SIZE),
                                                    batch_size=BATCH_SIZE)

validation_generator = test_datagen.flow_from_dataframe(validation_dataframe, directory="dataset/base_set",
                                                        x_col='filename', y_col="family", class_mode="categorical",
                                                        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE)
# input layer
input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))


x = Conv2D(filters=16,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='valid',
           input_shape=(IMG_SIZE, IMG_SIZE, 3),
           data_format='channels_last',
           activation='sigmoid')(input)
x = MaxPooling2D(pool_size=(2, 2),
                 strides=2)(x)
x = Dropout(rate=0.2)(x)

y = Conv2D(filters=64,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding='valid',
           activation='relu')(input)
y = MaxPooling2D(pool_size=(2, 2))(y)
y = Dropout(rate=0.2)(y)


x = concatenate([x, y])
x = MaxPooling2D(pool_size=(2, 2),
                 strides=2)(x)
x = Conv2D(filters=64, kernel_size=(8, 8), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2),
                 strides=2)(x)


x = Dense(64, activation='relu')(x)
x = Dropout(rate=0.2)(x)


x = Flatten()(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=input, outputs=output)

print(model.summary())

plot_model(model, to_file='network.png')

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=3,
    validation_data=validation_generator,
    validation_steps=2)

plot_training(history)

model.save_weights("task1_mark.h5")
