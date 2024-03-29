import re
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
def remove_stop_words(utterance):
    stop_words = list(stopwords.words('english'))
    stop_words += ['so','yeah','well','uh','ok','now','we','know','that','we','say','mean','this','think','guess',
                  'just','like','imagine','yes','here','there']
 
    word_tokens = word_tokenize(utterance)
    # converts the words in word_tokens to lower case and then checks whether 
    #they are present in stop_words or not
    filtered_utterance = [w for w in word_tokens if not w.lower() in stop_words]
    #with no lower case conversion
    filtered_utterance = []

    for w in word_tokens:
        if w not in stop_words:
            filtered_utterance.append(w)
    return " ".join(filtered_utterance)
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
    
def broaden_search_with_numbers(common_grounds, mentioned_numbers):
    # Filter common grounds to include those mentioning any of the mentioned numbers
    
    broadened_common_grounds = [cg for cg in common_grounds if any(str(number) in cg for number in mentioned_numbers)]
    
    return broadened_common_grounds
def broaden_search_with_colors(common_grounds, mentioned_colors):
    # Filter common grounds to include those mentioning any of the mentioned colors
    broadened_common_grounds = [cg for cg in common_grounds if any(color in cg for color in mentioned_colors)]
    return broadened_common_grounds
