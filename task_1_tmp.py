from keras import preprocessing as pr
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from pandas import DataFrame
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

dataframe = pd.read_csv("dataset/butterflies.txt", sep='\t')
df = dataframe.drop(columns=['species', 'genus', 'subfamily'])

df.loc[df['family'].isin([1]), 'family'] = "Papilionidae"
df.loc[df['family'].isin([2]), 'family'] = "Pieridae"
df.loc[df['family'].isin([3]), 'family'] = "Nymphalidae"
df.loc[df['family'].isin([4]), 'family'] = "Lycaenidae"
df.loc[df['family'].isin([5]), 'family'] = "Hesperiidae"

df = df.loc[df['family'].isin(["Pieridae", "Papilionidae"])]
print(df)

train_dataframe = pd.DataFrame.sample(df, frac=0.7)

print (train_dataframe)
validation_dataframe = pd.DataFrame.drop(df, train_dataframe.index)

train_datagen = pr.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = pr.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_dataframe(train_dataframe,
                                                    directory="dataset/base_set",
                                                    x_col='filename',
                                                    y_col="family", class_mode="categorical", target_size=(224, 224),
                                                    batch_size=8)

validation_generator = test_datagen.flow_from_dataframe(validation_dataframe,
                                                        directory="dataset/base_set",
                                                        x_col='filename',
                                                        y_col="family",
                                                        class_mode="categorical",
                                                        target_size=(224, 224),
                                                        batch_size=8)


# input layer
visible = Input(shape=(224,224,3))
# first feature extractor
conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
flat1 = Flatten()(pool1)
# second feature extractor
conv2 = Conv2D(16, kernel_size=8, activation='relu')(visible)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat2 = Flatten()(pool2)
# merge feature extractors
merge = concatenate([flat1, flat2])
# interpretation layer
hidden1 = Dense(10, activation='relu')(merge)
# prediction output
output = Dense(2, activation='sigmoid')(hidden1)
model = Model(inputs=visible, outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='shared_input_layer.png')


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    samples_per_epoch=60,
    epochs=10,
    validation_data=validation_generator,
    nb_val_samples=400)


exit()
model.save_weights("task1.h5")
