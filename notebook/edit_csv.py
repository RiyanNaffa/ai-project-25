import pandas as pd
import os
import matplotlib.pyplot as plt
import math

# Define the path to the data directory
data_dir = "../data/"

# List all files in the data directory
files = os.listdir(data_dir)
print(f"Files in {data_dir}: {files}\n\n")

cities = [file[:-4] for file in files if file.endswith('.csv')]
columns_to_drop = ['provider', 'owner_name', 'isMonitor', 'isMobile', 'country_iso']

for city in cities:
    print(f"========== City: {city} ==========\n")
    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(os.path.join(data_dir, f"{city}.csv"), parse_dates=['datetimeUtc'])
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        # 2024-07-08T12:00:00+07:00
        df['datetimeUtc'] = pd.to_datetime(df['datetimeUtc'])
        
        print(f"{city} parameter counts:\n", df['parameter'].value_counts(), "\n")
        print("Unique parameter: ", df['parameter'].unique())
        
        df_wide = df.pivot_table(index=['location_id', 'location_name', 'datetimeUtc', 'latitude', 'longitude'],
                                 columns='parameter', 
                                 values='value').reset_index()

        # Display basic information about the DataFrame
        print(f"Shape of the dataset: {df_wide.shape}")
        print("Columns in the dataset:", df_wide.columns)
        print("Data types:\n", df_wide.dtypes)
        print("Missing values:\n", df_wide.isnull().sum())
        print("Basic statistics:\n", df_wide.describe())
        
        # Display 5 first rows
        print("First 5 rows:\n", df_wide.head())
        print()
        
        # Set datetime as index for time series plots
        df_wide['datetimeUtc'] = pd.to_datetime(df_wide['datetimeUtc'])
        df_wide.set_index('datetimeUtc', inplace=False)
        
        data_processed_dir = "../data_processed/"
        os.makedirs(data_processed_dir, exist_ok=True)
        df_wide.to_csv(os.path.join(data_processed_dir, f"{city}_processed.csv"), index=False)

    except Exception as e:
        print(f"Could not process file {city}.csv: {e}")