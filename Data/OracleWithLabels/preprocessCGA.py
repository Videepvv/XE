import pandas as pd
import re

# Load the dataset
file_path = '03.csv'  # Replace with your file path
data = pd.read_csv(file_path)
data['Common Ground'] = data['Common Ground'].astype(str)
#print(data['Common Ground'])


#print(data)
# Filtering rows where 'Common Ground' contains 'STATEMENT'
statement_rows = data[data['Common Ground'].str.contains('STATEMENT')]
print(statement_rows)

# Define a function to extract content within brackets
def extract_brackets(text):
    match = re.search(r'\((.*?)\)', text)
    if match:
        return match.group(1)
    else:
        return "No Match Found"  # Return a placeholder if no match is found


# Apply the function to the 'Common Ground' column
statement_rows.loc[:, 'Common Ground'] = statement_rows['Common Ground'].apply(extract_brackets)
print(statement_rows)
# Save the modified dataframe back to the original file
#statement_rows.to_csv('new3.csv')
