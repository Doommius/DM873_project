from keras import preprocessing as pr
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from pandas import DataFrame


def get_subfamily(dataframe, subfamily):
    return dataframe(['subfamily' == 3])


def get_species(dataframe, species):
    return dataframe(['species' == 3])


def get_genus(dataframe, genus):
    return dataframe(['genus' == 3])


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

dataframe = pd.read_csv("dataset/butterflies.txt", sep='\t')
df: DataFrame = dataframe.drop(columns=['species', 'genus', 'subfamily'])

df.loc[df['family'].isin([1]), 'family'] = "Papilionidae"
df.loc[df['family'].isin([2]), 'family'] = "Pieridae"
df.loc[df['family'].isin([3]), 'family'] = "Nymphalidae"
df.loc[df['family'].isin([4]), 'family'] = "Lycaenidae"
df.loc[df['family'].isin([5]), 'family'] = "Hesperiidae"

print(df)


model = Sequential()
model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

train_dataframe = pd.DataFrame.sample(df, frac=0.8)
validation_dataframe = pd.DataFrame.drop(df, train_dataframe.index)

train_datagen = pr.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = pr.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_dataframe(train_dataframe, directory="dataset/base_set", x_col='filename',
                                                    y_col="family", class_mode="categorical", target_size=(32, 32),
                                                    batch_size=10)

validation_generator = test_datagen.flow_from_dataframe(validation_dataframe, directory="dataset/base_set",
                                                        x_col='filename', y_col="family", class_mode="categorical",
                                                        target_size=(32, 32), batch_size=10)

model.fit_generator(
    train_generator,
    samples_per_epoch=50,
    epochs=20,
    validation_data=validation_generator,
    nb_val_samples=50)

model.save_weights("task3.h5")