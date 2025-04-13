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
        df_wide.set_index('datetimeUtc', inplace=True)

        # Plot all pollutant parameters over time
        pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        available_pollutants = [col for col in pollutant_cols if col in df_wide.columns]

        # Save individual plots
        output_dir = "plots/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine all parameters you want to plot
        all_cols = available_pollutants
        n = len(all_cols)

        # Determine grid size for subplots (e.g., 2 columns)
        cols = 2
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), sharex=True)
        axes = axes.flatten()

        for i, col in enumerate(all_cols):
            axes[i].scatter(df_wide.index, df_wide[col], color='tab:blue')
            axes[i].set_title(f"{col.upper()} Over Time - {city}")
            axes[i].set_xlabel("Datetime (UTC)")
            axes[i].set_ylabel(f"{col} value")
            axes[i].grid(True)

        # Hide any extra empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        collage_filename = os.path.join(output_dir, f"{city}_pollutants_plot.png")
        plt.savefig(collage_filename)
        plt.close()

        print(f"Saved plot to: {collage_filename}")

    except Exception as e:
        print(f"Could not process file {city}.csv: {e}")