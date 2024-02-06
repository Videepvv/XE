import pandas as pd
import re
import re
def change_sybbols(dp_content):
    dp_content = dp_content.replace("equals"," = ").replace("does not equal"," != ").\
                        replace("is less than"," < ").replace("is more than"," > ").\
                        replace("plus"," + ").replace("ten"," 10 ").replace("twenty"," 20 ").replace("thirty"," 30 ").\
                        replace("forty"," 40 ").replace("fifty"," 50 ").replace("block","").replace("and", " , ") 
    
    dp_content = re.sub(r'\s+', ' ', dp_content)                   
    return dp_content.strip()

import pandas as pd


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


df = pd.read_csv('longListOfProps.txt', names = ['Propositions'])
print(df)
df['Propositions'] = df["Propositions"].apply(change_sybbols)
df.to_csv('correctedList.csv')
# df = pd.read_csv("correctedList.csv")
df['Propositions'] = [normalize_expression(expr) for expr in df['Propositions']]
df= df.drop_duplicates()
df.to_csv('NormalizedList.csv')


