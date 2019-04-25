import pandas as pd

data_set = pd.read_csv("dataset/butterflies.txt", sep='\t')
print(pd.DataFrame([data_set], index= family ).T)
