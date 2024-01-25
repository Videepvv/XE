import pandas as pd
import random

listOfDataFrames = []
for group in range(2,11):
    listOfDataFrames.append(pd.read_csv(f'Modified_{group:02d}.csv'))

dataset = pd.concat(listOfDataFrames) # this will hold all of the utterances and common ground
dataset['Label'] = 1


common_grounds_dataSet = (pd.read_csv('listOfPropositions.csv')) # This will have the list of all the possible propositions 
common_grounds = list(common_grounds_dataSet['Propositions'])
dataset = dataset[['Common Ground', 'Label', 'Transcript']]
# Duplicating rows with random common ground and label 0, creating 4 new instances for each row
new_rows = []
for index, row in dataset.iterrows():
    for _ in range(4):  # Repeat 4 times for each row
        new_row = row.copy()
        new_row['Common Ground'] = random.choice(common_grounds)
        new_row['Label'] = 0
        new_rows.append(new_row)

# Append new rows to the original dataframe
#df_extended = dataset.append(new_rows, ignore_index=True)
df_extended = pd.concat([dataset, pd.DataFrame(new_rows)], ignore_index=True)

df_extended.to_csv('Dataset.csv')
#print(dataset)