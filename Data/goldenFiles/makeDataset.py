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
    listOfDataFrames.append(pd.read_csv(f'Golden_Group_{group:02d}_CGA.csv'))

dataset = pd.concat(listOfDataFrames) # this will hold all of the utterances and common ground
dataset['Label'] = 1


common_grounds_dataSet = (pd.read_csv('listOfPropositions.csv')) # This will have the list of all the possible propositions 
common_grounds = list(common_grounds_dataSet['Propositions'])
dataset = dataset[['Common Ground', 'Label', 'Transcript']]
# Duplicating rows with random common ground and label 0, creating 4 new instances for each row
new_rows = []
for index, row in dataset.iterrows():
    transcript_colors = extract_colors(row['Transcript'])
    original_common_ground = row['Common Ground'] # This contatins the original common ground
    
    if transcript_colors:
        # Filter common grounds that contain any of the colors in the transcript
        filtered_common_grounds = [cg for cg in common_grounds if any(color in cg for color in transcript_colors) and cg != original_common_ground]
        
        # Create new rows for each filtered common ground
        for _ in range(4):  # Repeat 4 times for each row
            if filtered_common_grounds:  # Check if there are any filtered common grounds   
                selected_common_ground = random.choice(filtered_common_grounds)
                new_row = row.copy()
                new_row['Common Ground'] = selected_common_ground
                new_row['Label'] = 0
                new_rows.append(new_row)

# Append new rows to the original dataframe
df_extended = pd.concat([dataset, pd.DataFrame(new_rows)], ignore_index=True)

df_extended.to_csv('Dataset_Updated.csv')