import pandas as pd
import random
import re
from helperMethods import is_proposition_present, normalize_expression, \
normalize_sub_expression, extract_colors_and_numbers, is_valid_common_ground, is_valid_individual_match, remove_stop_words

def broaden_search_with_numbers(common_grounds, mentioned_numbers):
    # Filter common grounds to include those mentioning any of the mentioned numbers
    
    broadened_common_grounds = [cg for cg in common_grounds if any(str(number) in cg for number in mentioned_numbers)]
    
    return broadened_common_grounds
def broaden_search_with_colors(common_grounds, mentioned_colors):
    # Filter common grounds to include those mentioning any of the mentioned colors
    broadened_common_grounds = [cg for cg in common_grounds if any(color in cg for color in mentioned_colors)]
    return broadened_common_grounds

# def is_proposition_present(correct_proposition, filtered_common_grounds):
#     # Normalize the correct proposition to match the format of filtered common grounds
#     normalized_correct_proposition = normalize_expression(correct_proposition)
#     return normalized_correct_proposition in filtered_common_grounds

# def normalize_expression(expr):
#     # Split the expression into sub-expressions by commas for separate processing
#     sub_expressions = expr.split(',')
#     normalized_sub_expressions = [normalize_sub_expression(sub.strip()) for sub in sub_expressions]
#     # Sort the equalities if they involve simple color and number assignments
#     if all('=' in sub and (sub.strip().split('=')[0].strip().isalpha() and sub.strip().split('=')[1].strip().isdigit()) for sub in normalized_sub_expressions):
#         normalized_sub_expressions.sort()
#     return ', '.join(normalized_sub_expressions)

# def normalize_sub_expression(sub_expr):
#     # Identify the first operator and split the expression around it
#     match = re.search(r'([=!<>]+)', sub_expr)
#     if match:
#         operator = match.group(1)
#         parts = re.split(r'([=!<>]+)', sub_expr, 1)
#         left_side = parts[0].strip()
#         right_side = parts[2].strip()

#         # Normalize right side if there is a '+' or for '=', '!=' without '+'
#         if '+' in right_side:
#             right_side_components = re.findall(r'\w+', right_side)
#             right_side_sorted = ' + '.join(sorted(right_side_components))
#             return f"{left_side} {operator} {right_side_sorted}"
#         elif operator in ['=', '!=']:
#             # For '=' and '!=', sort operands alphabetically if no '+' on the right side
#             if not right_side.isdigit() and left_side > right_side:  # Avoid sorting number assignments
#                 return f"{right_side} {operator} {left_side}"
#             else:
#                 return sub_expr
#         else:
#             # For '<' and '>', return as is when no '+' on the right side
#             return sub_expr
#     else:
#         # Return the expression as is if it doesn't match the above conditions
#         return sub_expr
# # Mapping words to numbers for comparison

# number_mapping = {
#     "ten": 10, "twenty": 20, "thirty": 30, 
#     "forty": 40, "fifty": 50
# }

# def extract_colors_and_numbers(text):
#     colors = ["red", "blue", "green", "yellow", "purple"]
#     numbers = list(number_mapping.keys())
#     found_elements = {"colors": [], "numbers": []}
#     for color in colors:
#         if color in text:
#             found_elements["colors"].append(color)
#     for number in numbers:
#         if number in text:
#             found_elements["numbers"].append(number_mapping[number])
#     return found_elements


# def is_valid_common_ground(cg, elements):
#     cg_colors = re.findall(r'\b(?:red|blue|green|yellow|purple)\b', cg)
#     cg_numbers = [int(num) for num in re.findall(r'\b(?:10|20|30|40|50)\b', cg)]
#     #print(cg_colors, cg_numbers)
#     color_match = not elements["colors"] or set(cg_colors) == set(elements["colors"])
#     number_match = not elements["numbers"] or set(cg_numbers) == set(elements["numbers"])
#     return color_match and number_match

# def is_valid_individual_match(cg, elements):
#     cg_colors = re.findall(r'\b(?:red|blue|green|yellow|purple)\b', cg)
#     cg_numbers = [int(num) for num in re.findall(r'\b(?:10|20|30|40|50)\b', cg)]

#     for color in elements["colors"]:
#         for number in elements["numbers"]:
#             if color in cg_colors and number in cg_numbers:
#                 return True
#     return False
    

listOfDataFrames = []
for group in range(1, 11):
    listOfDataFrames.append(pd.read_csv(f'Golden_Group_{group:02d}_CGA.csv'))

dataset = pd.concat(listOfDataFrames)
#print(dataset.shape)
#dataset = pd.read_csv('updated.csv')
dataset['Label'] = 1
#dataset['Common Ground']= dataset['Common Ground'].replace("and", " , ")
common_grounds_dataSet = pd.read_csv('NormalizedList.csv')
common_grounds = list(common_grounds_dataSet['Propositions'])
dataset = dataset[['Common Ground', 'Label', 'Transcript', 'Group']]
listOfcommonGrounds = []
new_rows = []
total_transcripts = 0
propositions_lost = 0
rows_to_remove = []
for index, row in dataset.iterrows():
   
    elements = extract_colors_and_numbers(row['Transcript'].lower())
    original_common_ground = row['Common Ground'].replace("and", " , ") #raw common ground
    filtered_common_grounds = [cg for cg in common_grounds if is_valid_common_ground(cg, elements)] #list of possible common grounds
    
    original_common_ground = normalize_expression(original_common_ground) #normalize original
    filtered_common_grounds = [normalize_expression(expr) for expr in filtered_common_grounds] #normalize filtered
   
    

    if not filtered_common_grounds:  # If no match found, try individual color-number pairs
        filtered_common_grounds = [cg for cg in common_grounds if is_valid_individual_match(cg, elements)]
    #if(len(filtered_common_grounds)==5025 or len(filtered_common_grounds)==1 ):
    #if(len(filtered_common_grounds)==5025):
        #print("TRANSCRIPT - ", row['Transcript'].lower())
        #continue
    if(not elements['colors'] and not elements['numbers']):
        print("ORIGINAL COMMON GROUND - ", original_common_ground)
        print("TRANSCRIPT - ", row['Transcript'].lower())
        rows_to_remove.append(row['Transcript'])
        
        continue
    if not is_proposition_present(original_common_ground, filtered_common_grounds):
        
        #print("mentioned numbers - ", mentioned_numbers)
        mentioned_colors = elements['colors']
        filtered_common_grounds = broaden_search_with_colors(common_grounds, mentioned_colors)
        
        if not is_proposition_present(original_common_ground, filtered_common_grounds):
            # propositions_lost += 1
            # print("ORIGINAL COMMON GROUND - ", original_common_ground)
            # print("TRANSCRIPT - ", row['Transcript'].lower(), '\n')
            mentioned_numbers = elements['numbers']
            filtered_common_grounds = broaden_search_with_numbers(common_grounds, mentioned_numbers)
            if not is_proposition_present(original_common_ground, filtered_common_grounds):
                propositions_lost += 1
                print("ORIGINAL COMMON GROUND - ", original_common_ground)
                print("TRANSCRIPT - ", row['Transcript'].lower())
    #             print("mentioned numbers - ", mentioned_numbers)
                rows_to_remove.append(row['Transcript'])
                continue
    #if(row['Transcript'] == "I will tell you that the red cube is ten grams"):
        #print(filtered_common_grounds)
    total_transcripts += 1
    listOfcommonGrounds.append(len(filtered_common_grounds))
   
    filtered_common_grounds = [cg for cg in filtered_common_grounds if cg != original_common_ground]
    
    #listOfcommonGrounds.append(len(filtered_common_grounds))
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

print(df_extended.shape)
print(propositions_lost/total_transcripts)    
#df_extended = pd.concat([dataset, pd.DataFrame(new_rows)], ignore_index=True)
print(len(listOfcommonGrounds))
#print(sum(listOfcommonGrounds)/len(listOfcommonGrounds))
#print(listOfcommonGrounds)
df_extended = df_extended[~df_extended['Transcript'].isin(rows_to_remove)]
print(df_extended.shape)
#df_extended['Common Ground'] = [normalize_expression(expr) for expr in df_extended['Common Ground']]
#df_extended.to_csv('BigPrune_Dataset_Updated.csv')
df_extended['Transcript'] = df_extended['Transcript'].apply(remove_stop_words)
df_extended['Common Ground'] = df_extended['Common Ground'].str.replace("and"," , ")
df_extended['Common Ground'] = df_extended['Common Ground'].apply(normalize_expression)

df_extended.to_csv('preprocessedTrainingData.csv')