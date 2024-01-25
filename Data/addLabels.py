import pandas as pd
import re
# Function to find overlap between two intervals
def is_overlap(cgstart1, cgend1, ostart2, oend2):
    #return max(start1, start2) < min(end1, end2)
    return cgend1 >= ostart2 and cgstart1 <= oend2 
for group in range(2,11):
    print(group)
    # Load the Oracle and CGA CSV files
    oracle_path = f"Group_{group:02}_Oracle.csv"
    cga_path = f"Group_{group:02}_CGA.csv"

    oracle_df = pd.read_csv(oracle_path)
    cga_df = pd.read_csv(cga_path)

    # Convert time stamps to float
    oracle_df['Start'] = oracle_df['Start'].astype(float)
    oracle_df['End'] = oracle_df['End'].astype(float)
    cga_df['Begin Time - ss.msec'] = cga_df['Begin Time - ss.msec'].astype(float)
    cga_df['End Time - ss.msec'] = cga_df['End Time - ss.msec'].astype(float)

    # Create a new column in CGA for each Oracle column to be merged
    for col in oracle_df.columns:
        if col not in cga_df:
            cga_df[col] = None

    # Iterate through each row in CGA and check for overlap with Oracle rows
    for index, cga_row in cga_df.iterrows():
        cga_start = cga_row['Begin Time - ss.msec']
        cga_end = cga_row['End Time - ss.msec']

        for _, oracle_row in oracle_df.iterrows():
            oracle_start = oracle_row['Start']
            oracle_end = oracle_row['End']

            # Check for overlap
            if is_overlap(cga_start, cga_end, oracle_start, oracle_end):
                # Merge Oracle row contents into CGA row
                for col in oracle_df.columns:
                    cga_df.at[index, col] = oracle_row[col]

    #THE FOLLOWING WILL FILTER OUT ONLY THE STATEMENTS 
    # Save the modified CGA dataframe to a new CSV file
    modified_cga_path = f'OracleWithLabels/{group:02d}.csv'
    cga_df['Common Ground'] = cga_df['Common Ground'].astype(str)

    # Filtering rows where 'Common Ground' contains 'STATEMENT'
    statement_rows = cga_df[cga_df['Common Ground'].str.contains('STATEMENT')]
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
    statement_rows.to_csv(f'OracleWithLabels/testingNew/Modified_{group:02d}.csv')
    #cga_df.to_csv(modified_cga_path, index=False)
