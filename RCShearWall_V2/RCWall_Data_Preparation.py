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
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)
    elif file_type == 'parquet':
        df = pd.DataFrame(data)
        df.to_parquet(filename, index=False)


def process_data():
    # Read CSV file
    df = open_csv_file(f"{FOLDER}/{FILENAME}")
    data_points = split_rows(df)

    # Save Inputs & Outputs to both CSV and Parquet files
    for name, row_index in OUTPUT_FILES.items():
        data = extract_values(data_points, row_index)
        for file_format in FILE_FORMATS:
            output_file = f"{FOLDER}/Processed_Data/Data_30K/{name}.{file_format}"
            save_data(output_file, data, file_format)


if __name__ == "__main__":
    # Configuration
    FOLDER = "RCWall_Data"
    FILENAME = "Original_Data/Original_Data_30K.csv"
    FILE_FORMATS = ['csv', 'parquet']
    OUTPUT_FILES = {
        'InputParameters': 0,
        'InputCyclicDisplacement': 1,
        'OutputCyclicShear': 2
    }
    process_data()
    print("Processing completed successfully!")
