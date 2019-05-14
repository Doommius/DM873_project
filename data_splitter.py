'''
 Created by PyCharm.
 User: Mark Jervelund <mark@jervelund.com>
 Date: 14/05/19
 Time: 4:46 PM
'''

import pandas as pd
from pandas import DataFrame
import shutil
import os

# This is a simple script that splits the data into folders as required per the first assignment.
# This is not he correct way of doing it as it should be handle in software via dataframes and not via the file structure as this assignment requires.

## BEGIN unused code
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

dataframe = pd.read_csv("dataset/butterflies.txt", sep='\t')
#drop unused collums.
df = dataframe.drop(columns=['species', 'genus', 'subfamily'])
## END unused code


#task 1
task = "task1/a/"
# Delete old files
if os.path.exists(task) and os.path.isdir(task):
    shutil.rmtree(task)

os.mkdir(task)
os.mkdir(task+"/train/")
os.mkdir(task+"/validate/")

samples = ["Papilionidae", "Pieridae"]

for sample in samples:

    df.loc[df['family'].isin([2]), 'family'] = sample
    df_active_train = pd.DataFrame.sample(df, frac=0.8)
    df_active_validate = pd.DataFrame.drop(df, df_active_train.index)
    os.mkdir(task+"/train/"+sample)

    os.mkdir(task+"/validate/"+sample)

    for row in df_active_train.itertuples():
        shutil.copyfile("dataset/base_set/"+row[1], task+"train/"+sample+"/"+(row[1].replace("/","_")))

    for row in df_active_validate.itertuples():
        shutil.copyfile("dataset/base_set/"+row[1], task+"validate/"+sample+"/"+(row[1].replace("/","_")))

#task 2
task = "task2/b/"
samples = ["Nymphalidae", "Lycaenidae"]

# Delete old files
if os.path.exists(task) and os.path.isdir(task):
    shutil.rmtree(task)
os.mkdir(task)
os.mkdir(task+"/train/")
os.mkdir(task+"/validate/")


for sample in samples:

    df.loc[df['family'].isin([2]), 'family'] = sample
    df_active_train = pd.DataFrame.sample(df, n=250)
    df_active_validate = pd.DataFrame.drop(df, df_active_train.index)
    os.mkdir(task+"/train/"+sample)

    os.mkdir(task+"/validate/"+sample)

    for row in df_active_train.itertuples():
        shutil.copyfile("dataset/base_set/"+row[1], task+"train/"+sample+"/"+(row[1].replace("/","_")))

    for row in df_active_validate.itertuples():
        shutil.copyfile("dataset/base_set/"+row[1], task+"validate/"+sample+"/"+(row[1].replace("/","_")))

#task 3
task = "task3/c/"
# Delete old files
if os.path.exists(task) and os.path.isdir(task):
    shutil.rmtree(task)
os.mkdir(task)
os.mkdir(task+"/train/")
os.mkdir(task+"/validate/")
samples = ["Papilionidae","Pieridae","Nymphalidae","Lycaenidae","Hesperiidae"]
for sample in samples:

    df.loc[df['family'].isin([2]), 'family'] = sample
    df_active_train = pd.DataFrame.sample(df, frac=0.8)
    df_active_validate = pd.DataFrame.drop(df, df_active_train.index)
    os.mkdir(task+"/train/"+sample)

    os.mkdir(task+"/validate/"+sample)

    for row in df_active_train.itertuples():
        shutil.copyfile("dataset/base_set/"+row[1], task+"train/"+sample+"/"+(row[1].replace("/","_")))

    for row in df_active_validate.itertuples():
        shutil.copyfile("dataset/base_set/"+row[1], task+"validate/"+sample+"/"+(row[1].replace("/","_")))