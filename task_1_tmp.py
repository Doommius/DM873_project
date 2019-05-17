from keras import preprocessing as pr
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D
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



#load data from file to dataframe and change family to string and not int.
dataframe = pd.read_csv("dataset/butterflies.txt", sep='\t')
df = dataframe.drop(columns=['species', 'genus', 'subfamily'])

df.loc[df['family'].isin([1]), 'family'] = "Papilionidae"
df.loc[df['family'].isin([2]), 'family'] = "Pieridae"
df.loc[df['family'].isin([3]), 'family'] = "Nymphalidae"
df.loc[df['family'].isin([4]), 'family'] = "Lycaenidae"
df.loc[df['family'].isin([5]), 'family'] = "Hesperiidae"

df = df.loc[df['family'].isin(["Pieridae", "Papilionidae"])]
print(df)

##from data frame to data generator
train_dataframe = pd.DataFrame.sample(df, frac=0.7)
validation_dataframe = pd.DataFrame.drop(df, train_dataframe.index)

train_datagen = pr.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

test_datagen = pr.image.ImageDataGenerator()

train_generator = train_datagen.flow_from_dataframe(train_dataframe, directory  ="dataset/base_set", x_col='filename',
                                                    y_col="family", class_mode="categorical", target_size=(224, 224),
                                                    batch_size=64)

validation_generator = test_datagen.flow_from_dataframe(validation_dataframe, directory="dataset/base_set",
                                                        x_col='filename', y_col="family", class_mode="categorical",
                                                        target_size=(224, 224), batch_size=64)

##Network begin.
model = Sequential()
model.add(Conv2D(64, (3, 3), activation="relu", input_shape=(224, 224, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(224, 3, 3,activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, 3, 3,activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.9))
model.add(Dense(64,activation="relu"))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=3,
    validation_data=validation_generator,
    nb_val_samples=10)

plot_training(history)

model.save_weights("task1_tmp.h5")