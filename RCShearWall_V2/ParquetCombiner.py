import os
import pandas as pd
from tqdm import tqdm  # Import tqdm for the progress bar

# Define the input folders
folder_1 = r'K:\RCShearWall_V2\RCWall_Data\prepared\Run_Final_1\FullData'
folder_2 = r'K:\RCShearWall_V2\RCWall_Data\prepared\Run_Final_2\FullData'
folder_3 = r'K:\RCShearWall_V2\RCWall_Data\prepared\Run_Final_3\FullData'
folder_4 = r'K:\RCShearWall_V2\RCWall_Data\prepared\Run_Full2\FullData'

# Define the output folder
output_folder = r'K:\RCShearWall_V2\RCWall_Data\prepared\Run_Final_Full\FullData'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# List of file names to process
file_names = [
    "a1.parquet",
    "c1.parquet",
    "a2.parquet",
    "c2.parquet",
    "InputDisplacement.parquet",
    "InputParameters.parquet",
    "OutputShear.parquet"
]

# Loop through each file and stack with a progress bar
for file_name in tqdm(file_names, desc="Processing files", unit="file"):
    file_paths = [
        os.path.join(folder_1, file_name),
        os.path.join(folder_2, file_name),
        os.path.join(folder_3, file_name),
        os.path.join(folder_4, file_name),
    ]

    # Filter only existing files
    valid_file_paths = [path for path in file_paths if os.path.exists(path)]

    # Check if at least two files exist to stack
    if len(valid_file_paths) < 2:
        print(f"Skipping {file_name}: Less than two files found.")
        continue

    # Read and stack the DataFrames
    dfs = [pd.read_parquet(path) for path in valid_file_paths]
    df_stacked = pd.concat(dfs, ignore_index=True)

    # Save the stacked DataFrame to the output folder
    output_path = os.path.join(output_folder, file_name)
    df_stacked.to_parquet(output_path, index=False)

    print(f"Stacked file saved: {output_path}")