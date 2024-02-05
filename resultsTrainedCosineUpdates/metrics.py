import pandas as pd
import pandas as pd
import glob

# Step 1: List all CSV files in the current directory
csv_files = glob.glob('*_W.csv')

# Step 2: Read each CSV file and append it to a list
dfs = []  # Initialize an empty list to store DataFrames
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Step 3: Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Step 4: Save the combined DataFrame to a new CSV file
#combined_df.to_csv('combined_trainedcosine_results_csv.csv', index=False)
# Load the CSV file
file_path = 'combined_trainedcosine_results_csv.csv'  # Update this to your actual file path
data = combined_df
# Remove <m></m> tags from the actual_common_ground column and strip any whitespace
data['actual_common_ground_cleaned'] = data['actual_common_ground'].str.replace(r'<m>|</m>', '', regex=True).str.strip()

# Group the data by 'transcript'
grouped = data.groupby('transcript')

# Function to determine if the highest score within each transcript group matches the actual common ground
def is_actual_in_top_2_scores(group):
    # Sort the group by scores in descending order to get the top scored entries
    sorted_group = group.sort_values(by='scores', ascending=False)
    # Check if the actual common ground is among the top 2
    top_2_common_grounds = sorted_group['common_ground'].head(2).tolist()
    actual_common_ground = sorted_group['actual_common_ground_cleaned'].iloc[0]
    return actual_common_ground in top_2_common_grounds
# Apply the function to each group of transcripts and aggregate the results
results = grouped.apply(is_actual_in_top_2_scores)

# Count the total number of transcripts where the highest score matches the actual common ground
total_matches = results.sum()

# Print the results
print(f"Total matches: {total_matches} out of {data['transcript'].nunique()} transcripts")

def calculate_iou(common_ground, actual_common_ground):
    # Split by ',' and strip whitespace from each element
    common_ground_elements = set([element.strip() for element in common_ground.split(',')])
    actual_common_ground_elements = set([element.strip() for element in actual_common_ground.split(',')])
    # Calculate intersection and union
    intersection = common_ground_elements & actual_common_ground_elements
    union = common_ground_elements | actual_common_ground_elements
    # Calculate IoU
    iou = len(intersection) / len(union) if union else 0
    return iou

# Define a function to apply IoU calculation for the top score in each transcript group
def get_top_iou_score(group):
    # Sort the group by scores in descending order and select the top row
    top_row = group.sort_values(by='scores', ascending=False).iloc[0]
    # Calculate IoU score for the top match
    iou_score = calculate_iou(top_row['common_ground'], group['actual_common_ground_cleaned'].iloc[0])
    return iou_score

# Apply the IoU calculation function to each group of transcripts for the top score
iou_scores_top_1 = grouped.apply(get_top_iou_score)

# Now, iou_scores_top_1 contains the IoU score for the top match of each transcript
# You can aggregate these scores or further analyze them based on your needs

# Example: Print the average IoU score across all transcripts
average_iou_score = iou_scores_top_1.mean()

print(f"Average IoU Score for the top matches across all transcripts: {average_iou_score}")
