import pandas as pd
import random
import re
def extract_colors(text):
    # List of colors to look for
    colors = ["red", "blue", "green", "yellow", "purple"]
    found_colors = []
    for color in colors:
        if color in text:
            found_colors.append(color)
    return found_colors


listOfDataFrames = []
for group in range(1,11):
    listOfDataFrames.append(pd.read_csv(f'Golden_Group{group:02d}_CGA.csv'))

dataset = pd.concat(listOfDataFrames) # this will hold all of the utterances and common ground
dataset['Label'] = 1


common_grounds_dataSet = (pd.read_csv('listOfPropositions.csv')) # This will have the list of all the possible propositions 
common_grounds = list(common_grounds_dataSet['Propositions'])
dataset = dataset[['Common Ground', 'Label', 'Transcript']]
# Duplicating rows with random common ground and label 0, creating 4 new instances for each row
new_rows = []
for index, row in dataset.iterrows():
    transcript_colors = extract_colors(row['Transcript'])
    if transcript_colors:
        # Filter common grounds that contain any of the colors in the transcript
        filtered_common_grounds = [cg for cg in common_grounds if any(color in cg for color in transcript_colors)]
        
        # Create new rows for each filtered common ground
        for cg in filtered_common_grounds:
            new_row = row.copy()
            new_row['Common Ground'] = cg
            new_row['Label'] = 0
            new_rows.append(new_row)

# Append new rows to the original dataframe
df_extended = pd.concat([dataset, pd.DataFrame(new_rows)], ignore_index=True)

df_extended.to_csv('Dataset_Updated.csv')