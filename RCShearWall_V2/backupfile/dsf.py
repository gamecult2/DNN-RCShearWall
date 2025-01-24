import pandas as pd
import os
from tqdm import tqdm


def vertically_stack_parquet_files(input_files, output_csv):
    """
    Vertically stack rows from multiple parquet files into a single CSV
    with progress tracking

    Parameters:
    input_files (list): List of input parquet file paths
    output_csv (str): Path for the output CSV file
    """
    # Read all input files
    dataframes = []
    print("Reading input files...")
    for file in tqdm(input_files, desc="Loading Parquet Files"):
        # Check if file exists
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")

        # Read parquet file
        df = pd.read_parquet(file)

        # Add a column to identify the source file
        df['source_file'] = os.path.splitext(os.path.basename(file))[0]

        dataframes.append(df)

    # Validate that all input files have the same number of rows
    row_counts = [len(df) for df in dataframes]
    if len(set(row_counts)) > 1:
        raise ValueError("Input files must have the same number of rows. "
                         f"Row counts: {dict(zip(input_files, row_counts))}")

    # Prepare the vertical stacking
    combined_data = []
    num_rows = len(dataframes[0])

    print("Stacking rows vertically...")
    for row_idx in tqdm(range(num_rows), desc="Processing Rows"):
        for df in dataframes:
            # Create a row with data from each file for the same row index
            row_data = df.iloc[row_idx].to_dict()
            combined_data.append(row_data)

    # Create combined DataFrame
    combined_df = pd.DataFrame(combined_data)

    # Reorder columns to put source_file first
    cols = combined_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('source_file')))
    combined_df = combined_df[cols]

    # Write to CSV with progress tracking
    print("Writing output CSV...")
    with tqdm(total=1, desc="Saving CSV") as pbar:
        combined_df.to_csv(output_csv, index=False)
        pbar.update(1)

    print(f"\nVertically stacked CSV created: {output_csv}")
    print(f"Combined DataFrame shape: {combined_df.shape}")


# Example usage
input_files = [
    'RCWall_Data/Run_Full2/MonotonicData/InputParameters.parquet',
    'RCWall_Data/Run_Full2/MonotonicData/InputDisplacement.parquet',
    'RCWall_Data/Run_Full2/MonotonicData/OutputShear.parquet'
]
output_file = 'RCWall_Data/Run_Full2/MonotonicData/MonotonicData.csv'

# Call the function
vertically_stack_parquet_files(input_files, output_file)