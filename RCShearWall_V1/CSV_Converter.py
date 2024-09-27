import os
import pandas as pd

# Set the directory containing CSV files
csv_folder = 'RCWall_Data/Processed_Data'

# Set the directory where you want to save Parquet files
parquet_folder = 'RCWall_Data/Processed_Data'

# Ensure the output directory exists, if not create it
if not os.path.exists(parquet_folder):
    os.makedirs(parquet_folder)

# List all files in the CSV folder
csv_files = os.listdir(csv_folder)

# Loop through each CSV file and convert it to Parquet
for csv_file in csv_files:
    if csv_file.endswith('.csv'):
        csv_path = os.path.join(csv_folder, csv_file)
        parquet_file = os.path.splitext(csv_file)[0] + '.parquet'
        parquet_path = os.path.join(parquet_folder, parquet_file)

        # Read CSV and write to Parquet
        df = pd.read_csv(csv_path, header=None)
        df.to_parquet(parquet_path, index=False)

        print(f"Converted {csv_file} to {parquet_file}")
