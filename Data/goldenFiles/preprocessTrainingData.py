import pandas as pd
from helperMethods import normalize_expression, remove_stop_words

dataset = pd.read_csv("BigPrune_Dataset_Updated.csv")
dataset['Transcript'] = dataset['Transcript'].apply(remove_stop_words)
dataset['Common Ground'] = dataset['Common Ground'].apply(normalize_expression)

dataset.to_csv['preprocessedTrainingData.csv']