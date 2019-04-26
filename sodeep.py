from keras import models
from keras import layers
from keras import preprocessing as pr
from keras import optimizers
import pandas as pd


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
        super(MyModel, self).__init___()
        self.dense1 = layers.Dense(20, actication='relu')
        self.dense2 = layers.Dense(20, actication='relu')
        self.dense3 = layers.Dense(10, actication='softmax')

    def call(self, dataframe):
        train_dataframe = pd.DataFrame.sample(self, frac=0.8, random_state=200)
        validation_dataframe = pd.DataFrame.drop(train_dataframe.index)

        train_datagen = pr.image.ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = pr.image.ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_dataframe(train_dataframe)

        validation_generator = test_datagen.flow_from_dataframe(validation_dataframe)

        model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=800)

        x = self.dense1(input)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


dataframe = pd.read_csv("dataset/butterflies.txt", sep='\t')
model = MyModel
model.call(dataframe(['family' == 3]), MyModel)
model.fit()
