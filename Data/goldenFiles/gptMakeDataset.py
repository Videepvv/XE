import pandas as pd
import random
import re

def normalize_expression(expr):
    # Split the expression into sub-expressions by commas, if any, for separate processing
    sub_expressions = expr.split(',')
    normalized_sub_expressions = [normalize_sub_expression(sub.strip()) for sub in sub_expressions]
    # Sort the normalized sub-expressions to ensure consistent ordering
    normalized_sub_expressions.sort()
    return ', '.join(normalized_sub_expressions)

def normalize_sub_expression(sub_expr):
    # Identify all components (words and numbers) and operators
    components = re.findall(r'\w+|[=!<>]+', sub_expr)
    if len(components) == 3 and components[1] in ['=', '!=']:  # Simple equalities or inequalities
        # Sort the two elements for these cases
        if components[0] > components[2]:
            components[0], components[2] = components[2], components[0]
    elif len(components) > 3 and components[1] in ['=', '!=']:  # Complex expressions with operations
        # Sort elements on the right side of the expression if it's a complex expression
        if '+' in sub_expr:
            # Split the right side further by '+' and sort
            right_side = sorted(sub_expr.split(components[1])[1].replace(' ', '').split('+'))
            # Reassemble the expression with the sorted right side
            components = [components[0], components[1]] + ['+'.join(right_side)]
    return ' '.join(components)
# Mapping words to numbers for comparison
number_mapping = {
    "ten": 10, "twenty": 20, "thirty": 30, 
    "forty": 40, "fifty": 50
}

def extract_colors_and_numbers(text):
    colors = ["red", "blue", "green", "yellow", "purple"]
    numbers = list(number_mapping.keys())
    found_elements = {"colors": [], "numbers": []}
    for color in colors:
        if color in text:
            found_elements["colors"].append(color)
    for number in numbers:
        if number in text:
            found_elements["numbers"].append(number_mapping[number])
    return found_elements


def is_valid_common_ground(cg, elements):
    cg_colors = re.findall(r'\b(?:red|blue|green|yellow|purple)\b', cg)
    cg_numbers = [int(num) for num in re.findall(r'\b(?:10|20|30|40|50)\b', cg)]
    #print(cg_colors, cg_numbers)
    color_match = not elements["colors"] or set(cg_colors) == set(elements["colors"])
    number_match = not elements["numbers"] or set(cg_numbers) == set(elements["numbers"])
    return color_match and number_match

def is_valid_individual_match(cg, elements):
    cg_colors = re.findall(r'\b(?:red|blue|green|yellow|purple)\b', cg)
    cg_numbers = [int(num) for num in re.findall(r'\b(?:10|20|30|40|50)\b', cg)]

    for color in elements["colors"]:
        for number in elements["numbers"]:
            if color in cg_colors and number in cg_numbers:
                return True
    return False
    

listOfDataFrames = []
for group in range(1, 11):
    listOfDataFrames.append(pd.read_csv(f'Golden_Group_{group:02d}_CGA.csv'))

dataset = pd.concat(listOfDataFrames)
print(dataset.shape)
dataset['Label'] = 1
#dataset['Common Ground']= dataset['Common Ground'].replace("and", " , ")
common_grounds_dataSet = pd.read_csv('correctedList.csv')
common_grounds = list(common_grounds_dataSet['Propositions'])
dataset = dataset[['Common Ground', 'Label', 'Transcript', 'Group']]
listOfcommonGrounds = []
new_rows = []
for index, row in dataset.iterrows():
    elements = extract_colors_and_numbers(row['Transcript'].lower())
    original_common_ground = row['Common Ground'].replace("and", " , ") #raw common ground
    filtered_common_grounds = [cg for cg in common_grounds if is_valid_common_ground(cg, elements)] #list of possible common grounds
    
    original_common_ground = normalize_expression(original_common_ground) #normalize original
    filtered_common_grounds = [normalize_expression(expr) for expr in filtered_common_grounds] #normalize filtered
    if not filtered_common_grounds:  # If no match found, try individual color-number pairs
        filtered_common_grounds = [cg for cg in common_grounds if is_valid_individual_match(cg, elements)]
    if(len(filtered_common_grounds)==1650 or len(filtered_common_grounds)==1 ):
        #print(row['Transcript'])
        continue
    
    
    
    listOfcommonGrounds.append(len(filtered_common_grounds))
   
    filtered_common_grounds = [cg for cg in filtered_common_grounds if cg != original_common_ground]
    
    listOfcommonGrounds.append(len(filtered_common_grounds))
    selected_common_grounds = set(filtered_common_grounds)  # Keep track of selected common grounds to avoid repeats
    for _ in range(4):
        if not selected_common_grounds:  # If no unique common ground left, choose randomly from all common grounds
            selected_common_ground = normalize_expression(random.choice(common_grounds))
        else:
            selected_common_ground = selected_common_grounds.pop() # Add to the set to avoid future selection
        new_row = row.copy()
        new_row['Common Ground'] = selected_common_ground
        new_row['Label'] = 0
        new_rows.append(new_row)
        df_extended = pd.concat([dataset, pd.DataFrame(new_rows)], ignore_index=True)


    
#df_extended = pd.concat([dataset, pd.DataFrame(new_rows)], ignore_index=True)

print(sum(listOfcommonGrounds)/len(listOfcommonGrounds))
print(listOfcommonGrounds)
df_extended.to_csv('Testing_Dataset_Updated.csv')
