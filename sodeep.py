from keras import models
from keras import layers
from keras import optimizers

import pandas as pd


#
#
#  Idea 1,
#  Parse the CSV file with all the information add the pictures to a array or data structure and fetch them from there?
#
#
class datafetcher():
    __data_set = ""

    def __init__(self, folder):
        self.folder = folder
        # load data into type
        self.__data_set = pd.read_csv("dataset/butterflies.txt", sep='\t')

    def get_family(family):
        return dataframe[self.__data_set['family'] == family]

    def get_subfamily(subfamily):
        return dataframe[self.__data_set['subfamily'] == subfamily]

    def get_speices(species):
        return dataframe[self.__data_set['species'] == species]

    def get_genus(genus):
        return dataframe[self.__data_set['genus'] == genus]


class MyModel(keras.model):
    def __init__(self):
        super(MyModel, self).__init___()
        self.dense1 = layers.Dense(20, actication='relu')
        self.dense2 = layers.Dense(20, actication='relu')
        self.dense3 = layers.Dense(10, actication='softmax')

    def call(self, dataframe):

        train_dataframe = dataframe.sample(frac=0.8, random_state=200)
        validation_dataframe = dataframe.drop(train.index)

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_dataframe(train_dataframe)

        validation_generator = test_datagen.flow_from_dataframe(validation_dataframe)

        model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=800)


        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


model = MyModel
model.call(datafetcher.get_family(2))
model.fit()
