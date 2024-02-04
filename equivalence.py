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
print(normalize_expression("blue = red"))  # Simple equalities
print(normalize_expression("red = blue"))  # Expected to match the previous output
print(normalize_expression("red != blue"))  # Simple inequalities
print(normalize_expression("blue != red"))  # Expected to match the previous output
print(normalize_expression("green = red + blue + purple + yellow"))  # Complex expressions
print(normalize_expression("green = blue + red + purple + yellow"))  # Expected to match the previous output
print(normalize_expression("green != blue + purple"))  # Inequality with multiple elements
print(normalize_expression("green != purple + blue"))  # Expected to match the previous output
print(normalize_expression("blue = 50 , green = 30"))  # Expressions with numbers
print(normalize_expression("green = 30, blue = 50"))  
print(normalize_expression("green != red + blue + purple + yellow"))  # Expressions with numbers
print(normalize_expression("green != yellow + blue + purple + red"))  

"""
def normalize_expression(expr):
    # Detect operators and split the expression accordingly, including whitespace around operators
    parts = re.split(r'(\s*[=!<>]+\s*)', expr)
    if len(parts) < 3:  # If the expression doesn't match the expected format
        return expr  # Return the original expression if no operator is found

    left_part, operator, right_part = parts[0].strip(), parts[1].strip(), parts[2].strip()
    
    # For '=', '!=', check if we are dealing with simple or complex expressions
    if operator in ['=', '!=']:
        left_elements = re.findall(r'\w+', left_part)
        right_elements = re.findall(r'\w+', right_part)

        # For simple comparisons (single elements on each side), sort the two elements
        if len(left_elements) == 1 and len(right_elements) == 1:
            sorted_elements = sorted([left_part, right_part])
            normalized_expr = f"{sorted_elements[0]} {operator} {sorted_elements[1]}"
        else:
            # For complex expressions, sort elements within each side
            normalized_left = '+'.join(sorted(left_elements))
            normalized_right = '+'.join(sorted(right_elements))
            normalized_expr = f"{normalized_left} {operator} {normalized_right}"
    else:
        # For '<' and '>', sort elements within each side but treat sides independently
        normalize_left = '+'.join(sorted(re.findall(r'\w+', left_part)))
        normalize_right = '+'.join(sorted(re.findall(r'\w+', right_part)))
        normalized_expr = f"{normalize_left} {operator} {normalize_right}"

    return normalized_expr

# Testing the revised function with various cases
print(normalize_expression("blue = red"))  # Expected to correctly normalize simple equalities
print(normalize_expression("red = blue"))  # Expected to match the previous output
print(normalize_expression("red != blue"))  # Expected to correctly normalize simple inequalities
print(normalize_expression("blue != red"))  # Expected to match the previous output
print(normalize_expression("green = red + blue + purple + yellow"))  # Handles complex expressions correctly
print(normalize_expression("green = blue + red + purple + yellow"))  # Handles complex expressions correctly

"""