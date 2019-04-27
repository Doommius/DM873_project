from keras import models
from keras import layers
from keras import preprocessing as pr
from keras import optimizers
from keras.optimizers import SGD
import tensorflow as tf
import pandas as pd

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense


#
#
#  Idea 1,
#  Parse the CSV file with all the information add the pictures to a array or data structure and fetch them from there?
#
#

def get_subfamily(datagrame, subfamily):
    return dataframe(['subfamily' == 3])


def get_species(datagrame, species):
    return dataframe(['species' == 3])


def get_genus(datagrame, genus):
    return dataframe(['genus' == 3])


class MyModel(models.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = layers.Dense(20, actication='relu')
        self.dense2 = layers.Dense(20, actication='relu')
        self.dense3 = layers.Dense(10, actication='softmax')

    def call(self, dataframe):


        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])


        train_dataframe = pd.DataFrame.sample(dataframe, frac=0.8)
        validation_dataframe = pd.DataFrame.drop(dataframe, train_dataframe.index)

        train_datagen = pr.image.ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = pr.image.ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_dataframe(train_dataframe,x_col='filename', y_col="family")

        validation_generator = test_datagen.flow_from_dataframe(validation_dataframe,x_col='filename', y_col="family")

        model.fit_generator(
            train_generator,
            samples_per_epoch=50,
            epochs=5,
            validation_data=validation_generator,
            nb_val_samples=50)

        x = self.dense1(input)
        x = self.dense2(x)
        x = self.dense3(x)
        return x



#config
#   pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


dataframe = pd.read_csv("dataset/butterflies.txt", sep='\t')


df = dataframe.drop(columns=['species', 'genus','subfamily'])
df = df.loc[df['family'].isin([1, 2])]
print(df)

model = applications.VGG16(weights='imagenet', include_top=False)

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))



model.call(model, df[df.family == 3])
model.fit()
