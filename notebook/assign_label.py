import pandas as pd
import os

# Example: Load your dataset (assuming it's in a CSV file)
df = pd.read_csv('../data_merged/merged_cleaned_data.csv')

# Threshold setiap polutan selama 24 jam
# berdasarkan panduan WHO (Air Quality Guidelines)
thresholds = {
    'pm25': 15,
    'pm10': 45,
    'no2': 25,
    'o3': 100,
    'so2': 40,
}

# Label dictionary
labels = {
    'Normal': 0,
    'Elevated': 1,
    'Pollution Spike': 2
}

# Function to compute a simple composite label
def assign_aq_label(row, thresholds):
    score = 0
    # Increase the score for every pollutant above its threshold
    for pollutant, thresh in thresholds.items():
        if row[pollutant] > thresh:
            score += 1
    # Define rule-based classes (customize these rules based on your domain or exploratory results)
    if score == 0:
        return labels['Normal']
    elif 0 < score <= 2:
        return labels['Elevated']
    else:
        return labels['Pollution Spike']

# Apply the function row-wise to create a new column for our target
df['aq'] = df.apply(lambda row: assign_aq_label(row, thresholds), axis=1)

# Optional: Save the merged DataFrame to a CSV file
data_labeled_dir = os.path.abspath("../data_merged/")
output_path = os.path.join(data_labeled_dir, "merged_cleaned_data.csv")
df.to_csv(output_path, index=False)
print(f"Labeled data saved to {output_path}")