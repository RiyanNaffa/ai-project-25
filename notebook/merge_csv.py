import pandas as pd
import os

# Define the path to the data directory
data_dir = os.path.abspath("../data_processed/")

# List all CSV files in the data directory
files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
print(f"CSV files in {data_dir}: {files}\n")

# Columns to drop
columns_to_drop = ['location_id', 'location_name', 'datetimeUtc', 'latitude', 'longitude', 'relativehumidity', 'temperature']
pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2']

cities = ['Hanoi', 'London', 'Madrid', 'Napoli', 'Paris']

# Initialize an empty DataFrame for merging with columns set to pollutant_cols
df_merged = pd.DataFrame(columns=pollutant_cols)

# Process each file
for file in files:
    if file[:-14] not in cities:
        continue
    try:        
        file_path = os.path.join(data_dir, file)
        df_wide = pd.read_csv(file_path, parse_dates=['datetimeUtc'])
        
        # Drop specified columns
        df_wide.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        # Align columns and concatenate to the merged DataFrame, filling missing values with NaN
        df_wide = df_wide.reindex(columns=pollutant_cols, fill_value=pd.NA)
        df_merged = pd.concat([df_merged, df_wide], ignore_index=True)
        
    except Exception as e:
        print(f"Could not process file {file}: {e}")

# Optional: Save the merged DataFrame to a CSV file
data_merged_dir = os.path.abspath("../data_merged/")
output_path = os.path.join(data_merged_dir, "merged_data.csv")
df_merged.to_csv(output_path, index=False)
print(f"Merged data saved to {output_path}")