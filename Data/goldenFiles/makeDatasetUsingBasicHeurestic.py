import pandas as pd
import random
import re
from helperMethods import is_proposition_present, normalize_expression, normalize_sub_expression, extract_colors
def is_proposition_present(correct_proposition, filtered_common_grounds):
    # Normalize the correct proposition to match the format of filtered common grounds
    normalized_correct_proposition = normalize_expression(correct_proposition)
    return normalized_correct_proposition in filtered_common_grounds

def normalize_expression(expr):
    # Split the expression into sub-expressions by commas for separate processing
    sub_expressions = expr.split(',')
    normalized_sub_expressions = [normalize_sub_expression(sub.strip()) for sub in sub_expressions]
    # Sort the equalities if they involve simple color and number assignments
    if all('=' in sub and (sub.strip().split('=')[0].strip().isalpha() and sub.strip().split('=')[1].strip().isdigit()) for sub in normalized_sub_expressions):
        normalized_sub_expressions.sort()
    return ', '.join(normalized_sub_expressions)

def normalize_sub_expression(sub_expr):
    # Identify the first operator and split the expression around it
    match = re.search(r'([=!<>]+)', sub_expr)
    if match:
        operator = match.group(1)
        parts = re.split(r'([=!<>]+)', sub_expr, 1)
        left_side = parts[0].strip()
        right_side = parts[2].strip()

        # Normalize right side if there is a '+' or for '=', '!=' without '+'
        if '+' in right_side:
            right_side_components = re.findall(r'\w+', right_side)
            right_side_sorted = ' + '.join(sorted(right_side_components))
            return f"{left_side} {operator} {right_side_sorted}"
        elif operator in ['=', '!=']:
            # For '=' and '!=', sort operands alphabetically if no '+' on the right side
            if not right_side.isdigit() and left_side > right_side:  # Avoid sorting number assignments
                return f"{right_side} {operator} {left_side}"
            else:
                return sub_expr
        else:
            # For '<' and '>', return as is when no '+' on the right side
            return sub_expr
    else:
        # Return the expression as is if it doesn't match the above conditions
        return sub_expr
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


common_grounds_dataSet = (pd.read_csv('correctedList.csv')) # This will have the list of all the possible propositions 
common_grounds = list(common_grounds_dataSet['Propositions'])
dataset = dataset[['Common Ground', 'Label', 'Transcript','Group']]
# Duplicating rows with random common ground and label 0, creating 4 new instances for each row
new_rows = []
total_transcripts = 0
propositions_lost = 0

for index, row in dataset.iterrows():
    transcript_colors = extract_colors(row['Transcript'])
    original_common_ground = row['Common Ground'].replace("and", " , ").strip() # This contatins the original common ground
    
    if transcript_colors:
        total_transcripts+=1
        # Filter common grounds that contain any of the colors in the transcript
        filtered_common_grounds = [cg for cg in common_grounds if any(color in cg for color in transcript_colors)]# and cg != original_common_ground]
        filtered_common_grounds = [normalize_expression(expr) for expr in filtered_common_grounds] #normalize filtered
        original_common_ground = normalize_expression(original_common_ground)
        if not is_proposition_present(original_common_ground, filtered_common_grounds):
            propositions_lost += 1
            print('original - ', original_common_ground)
            print('transcript -',  row['Transcript'])
            
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
print(propositions_lost/total_transcripts)
#df_extended.to_csv('Dataset_Updated.csv')