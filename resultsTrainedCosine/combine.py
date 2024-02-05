import pandas as pd
import glob

# Step 1: List all CSV files in the current directory
csv_files = glob.glob('*.csv')

# Step 2: Read each CSV file and append it to a list
dfs = []  # Initialize an empty list to store DataFrames
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Step 3: Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Step 4: Save the combined DataFrame to a new CSV file
combined_df.to_csv('combined_trainedcosine_results_csv.csv', index=False)
