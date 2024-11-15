import csv
import pandas as pd
import os
from tqdm import tqdm


def open_csv_file(filename):
    """Opens a CSV file and returns a list of rows, where each row is a list of values."""
    with open(filename, "r") as f:
        return list(csv.reader(f))


def split_rows(rows, batch_row=3):
    """Splits a list of rows into six lists, where each list contains the rows for a single data point."""
    return [rows[i:i + batch_row] for i in range(0, len(rows), batch_row)]


def extract_values(data_points, row_index):
    """Extracts values from a specific row of each data point and converts them to floats."""
    return [[float(value) for value in data_point[row_index][0:]] for data_point in data_points]


def save_data(filename, data, file_type='csv'):
    """Saves data to a file in the specified format (CSV or Parquet)."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure the directory exists
    if file_type == 'csv':
        pd.DataFrame(data).to_csv(filename, index=False, header=False)
    elif file_type == 'parquet':
        pd.DataFrame(data).to_parquet(filename, index=False)


def process_data(folder, filename, output_files, file_formats, processed_data_dir):
    # Read CSV file
    data_points = split_rows(open_csv_file(os.path.join(folder, filename)))

    # Save Inputs & Outputs to both CSV and Parquet files
    for name, row_index in output_files.items():
        data = extract_values(data_points, row_index)
        for file_format in file_formats:
            output_file = os.path.join(folder, processed_data_dir, f"{name}.{file_format}")
            save_data(output_file, data, file_format)


if __name__ == "__main__":
    # Configuration
    FOLDER = "RCWall_Data"
    FILENAME = "Original_Data/Filtered_Data.csv"
    FILE_FORMATS = ['csv', 'parquet']
    OUTPUT_FILES = {
        'InputParameters': 0,
        'InputCyclicDisplacement': 1,
        'OutputCyclicShear': 2
    }
    PROCESSED_DATA_DIR = "Processed_Data/Data"
    process_data(FOLDER, FILENAME, OUTPUT_FILES, FILE_FORMATS, PROCESSED_DATA_DIR)
    print("Processing completed successfully!")
