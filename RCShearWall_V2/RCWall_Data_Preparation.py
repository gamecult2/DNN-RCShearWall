import csv
import pandas as pd
import os
from tqdm import tqdm


def open_csv_file(filename):
    """Opens a CSV file and returns a list of rows, where each row is a list of values."""
    with open(filename, "r") as f:
        return list(csv.reader(f))


def split_rows(rows, batch_row=7):
    """Splits a list of rows into six lists, where each list contains the rows for a single data point."""
    return [rows[i:i + batch_row] for i in range(0, len(rows), batch_row)]


# def extract_values(data_points, row_index):
#     """Extracts values from a specific row of each data point and converts them to floats."""
#     return [[float(value) for value in data_point[row_index][0:]] for data_point in data_points]


def extract_values(data_points, row_index):
    """
    Extracts values from the specified row index of each batch and converts them to floats.
    Handles the specific column ranges for each row type within the batches.
    """
    # Define column ranges based on row position within each batch
    column_ranges = {
        0: (0, 17),  # First row of each batch: columns 0-16
        1: (0, 500),  # Second row of each batch: columns 0-499
        2: (0, 500),  # Third row of each batch: columns 0-499
        3: (0, 168),  # Fourth row of each batch: columns 0-167
        4: (0, 168),  # Fifth row of each batch: columns 0-167
        5: (0, 168),  # Sixth row of each batch: columns 0-167
        6: (0, 168)  # Seventh row of each batch: columns 0-167
    }

    try:
        start_col, end_col = column_ranges[row_index]
        result = []

        # Process each batch
        for batch in data_points:
            try:
                # Extract and convert values for the specified row in this batch
                row_values = [float(value) for value in batch[row_index][start_col:end_col]]
                result.append(row_values)
            except (IndexError, ValueError) as e:
                print(f"Error processing batch at row {row_index}: {str(e)}")
                continue

        return result
    except Exception as e:
        print(f"Error extracting values: {str(e)}")
        raise


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
    for name, row_index in tqdm(output_files.items(), desc="Processing data points"):
        data = extract_values(data_points, row_index)
        for file_format in file_formats:
            output_file = os.path.join(processed_data_dir, f"{name}.{file_format}")
            save_data(output_file, data, file_format)


if __name__ == "__main__":
    # Print folder selection options
    print("Select the folder to process:")
    print("1 - Full Data")
    print("2 - Cyclic Data")
    print("3 - Monotonic Data")

    # Take user input for folder selection
    choice = input("Enter your choice (1, 2, or 3): ").strip()

    # Set folder and filename based on user's choice
    if choice == '1':
        FOLDER = "RCWall_Data/OriginalData/Run_Full/FullData"
        FILENAME = "Full_Data.csv"
        PROCESSED_DATA_DIR = "RCWall_Data/Run_Full/FullData"
    elif choice == '2':
        FOLDER = "RCWall_Data/OriginalData/Run_Full/CyclicData"
        FILENAME = "Cyclic_Data.csv"
        PROCESSED_DATA_DIR = "RCWall_Data/Run_Full/CyclicData"
    elif choice == '3':
        FOLDER = "RCWall_Data/OriginalData/Run_Full/MonotonicData"
        FILENAME = "Monotonic_Data.csv"
        PROCESSED_DATA_DIR = "RCWall_Data/Run_Full/MonotonicData"
    else:
        print("Invalid choice. Please choose 1, 2, or 3.")
        exit()

    FILE_FORMATS = ['csv', 'parquet']
    OUTPUT_FILES = {
        'InputParameters': 0,
        'InputCyclicDisplacement': 1,
        'OutputCyclicShear': 2,
        'c1': 3,
        'a1': 4,
        'c2': 5,
        'a2': 6
    }

    process_data(FOLDER, FILENAME, OUTPUT_FILES, FILE_FORMATS, PROCESSED_DATA_DIR)
    print("Processing completed successfully!")
