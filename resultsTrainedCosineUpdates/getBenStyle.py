import pandas as pd
from collections import defaultdict

# Load the provided CSV files
model_metrics_path = 'modelMetrics.csv'
#dp_bert_10_path = '/path/to/DP_bert_10.csv'

model_metrics_df = pd.read_csv(model_metrics_path)
#dp_bert_10_df = pd.read_csv(dp_bert_10_path)

# Group by 'transcript' and sort within each group by 'scores' in descending order
grouped = model_metrics_df.groupby('transcript', as_index=False).apply(lambda x: x.sort_values('scores', ascending=False)).reset_index(drop=True)

# Initialize a dictionary to hold the transformed data
transformed_data = defaultdict(list)

# Iterate through each group to transform data
for name, group in grouped.groupby('transcript'):
    transformed_data['Transcript'].append(name)
    transformed_data['Ground truth'].append(group['actual_common_ground'].iloc[0])
    
    # Update to handle cases where the rank of the actual common ground is not found
    correct_rank = group[group['common_ground'] == group['actual_common_ground'].iloc[0]].index.min() - group.index.min() + 1
    transformed_data['Correct'].append(correct_rank if not pd.isna(correct_rank) else 0)
    
    for i in range(1, 11):  # Assuming we need top 10 propositions
        if i <= len(group):
            prop = group.iloc[i-1]['common_ground']
            score = group.iloc[i-1]['scores']
            transformed_data[f'Top {i} prop'].append(prop)
            transformed_data[f'Top {i} cosim'].append(score)
        else:
            transformed_data[f'Top {i} prop'].append('N/A')
            transformed_data[f'Top {i} cosim'].append('N/A')

# Convert the dictionary to a DataFrame
transformed_df = pd.DataFrame(transformed_data)

# Correcting the approach for handling cases where the actual common ground rank is not found
# If the actual common ground does not match any proposition, set the 'Correct' column to 0
transformed_df['Correct'] = transformed_df.apply(lambda x: 0 if x['Correct'] == len(group) + 1 else x['Correct'], axis=1)

# Save the transformed data to a new CSV file
transformed_csv_path = 'transformed_modelMetrics_to_DP_format_updated.csv'
transformed_df.to_csv(transformed_csv_path, index=False)

print(f"Transformed file saved to: {transformed_csv_path}")
