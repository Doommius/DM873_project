'''
 Created by PyCharm.
 User: Mark Jervelund <mark@jervelund.com>
 Date: 14/05/19
 Time: 4:46 PM

 Description:

 This is a simple script that splits the data into folders as required per the first assignment.
 This is not he correct way of doing it as it should be handle in software via dataframes and not via the file structure
 as this assignment requires it.


'''

import pandas as pd
from pandas import DataFrame
import shutil
import os

## Begin debugging code
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

dataframe = pd.read_csv("dataset/butterflies.txt", sep='\t')
# drop unused collums.
df = dataframe.drop(columns=['species', 'genus', 'subfamily'])

df.loc[df['family'].isin([1]), 'family'] = "Papilionidae"
df.loc[df['family'].isin([2]), 'family'] = "Pieridae"
df.loc[df['family'].isin([3]), 'family'] = "Nymphalidae"
df.loc[df['family'].isin([4]), 'family'] = "Lycaenidae"
df.loc[df['family'].isin([5]), 'family'] = "Hesperiidae"

## END debugging code.model.add(Dense(64).Activation('relu'))


'''
Function for splitting the data. part is the part of the task, eg, a,b,c in the case, samples is a list of that samples we want

@n is the number of samples we want in our training set.

@frac is the ration of training /validation we want

Both can be left blank and it uses 80/20 for training / validation.

'''



def sample_function(part, samples, n=None, frac=None):
    os.mkdir(part)
    os.mkdir(part + "/train/")
    os.mkdir(part + "/validate/")

    for sample in samples:
        df_tmp = df.loc[df['family'] == sample]

        print (df_tmp)
        if n is not None:
            df_train = pd.DataFrame.sample(df_tmp, n=n)
        elif frac is not None:
            df_train = pd.DataFrame.sample(df_tmp, frac=frac)
        else:
            df_train = pd.DataFrame.sample(df_tmp, frac=0.7)

        df_validate = pd.DataFrame.drop(df_tmp, df_train.index)
        os.mkdir(part + "/train/" + sample)

        os.mkdir(part + "/validate/" + sample)

        for row in df_train.itertuples():
            shutil.copyfile("dataset/base_set/" + row[1], part + "train/" + sample + "/" + (row[1].replace("/", "_")))

        for row in df_validate.itertuples():
            shutil.copyfile("dataset/base_set/" + row[1],
                            part + "validate/" + sample + "/" + (row[1].replace("/", "_")))


task = "task1"
if os.path.exists(task) and os.path.isdir(task):
    shutil.rmtree(task)
os.mkdir(task)

# task 1
part = task + "/a/"
samples = ["Papilionidae", "Pieridae"]
sample_function(part, samples, frac=0.7)

# task 2
part = task + "/b/"
samples = ["Nymphalidae", "Lycaenidae"]
sample_function(part, samples, n=250)

# task 3
part = task + "/c/"
samples = ["Papilionidae", "Pieridae", "Nymphalidae", "Lycaenidae", "Hesperiidae"]
sample_function(part, samples, frac=0.7)
