import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score
# Load the uploaded file
file_path = '/s/babbage/b/nobackup/nblancha/public-datasets/ilideep/XE/resultsTrainedCosineUpdates/combined_trainedcosine_results_csv.csv'
df = pd.read_csv(file_path)

# Remove <m> and </m> tags from "actual common ground"
df["actual_common_ground"] = df["actual_common_ground"].str.replace("<m>", "").str.replace("</m>", "").str.strip()

# Add "true label" column; initially, set all to 0 to later update based on condition
df["true label"] = 0

# Ensure "common_ground" column has no leading/trailing spaces for accurate comparison
df["common_ground"] = df["common_ground"].str.strip()
df["true label"] = (df["actual_common_ground"] == df["common_ground"]).astype(int)
df["predicted label"] = (df["scores"] > 0.5).astype(int)
df.to_csv('modelMetrics.csv')


grouped_uploaded = df.groupby('transcript').agg({'true label': 'max'})
count_transcripts_no_true_common_ground_uploaded = (grouped_uploaded['true label'] == 0).sum()
transcripts_no_match = grouped_uploaded[grouped_uploaded['true label'] == 0].index.tolist()
df_cleaned = df[~df['transcript'].isin(transcripts_no_match)]

accuracy = accuracy_score(df_cleaned["true label"], df_cleaned["predicted label"])
print('acc' ,accuracy)
print('f1' , f1_score(df_cleaned["true label"], df_cleaned["predicted label"]))
# First, identify the transcripts with no true common ground match
transcripts_no_match = grouped_uploaded[grouped_uploaded['true label'] == 0].index.tolist()

# Filter the original DataFrame to get rows where 'transcript' is in the list of transcripts with no match
# and select 'transcript' and 'common_ground' columns
transcripts_no_match_details = df[df['transcript'].isin(transcripts_no_match)][['transcript', 'common_ground', 'actual_common_ground']].drop_duplicates()
transcripts_no_match_details.to_csv('cosine_failed.csv')