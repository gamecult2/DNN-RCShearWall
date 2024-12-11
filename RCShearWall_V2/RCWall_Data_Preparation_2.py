import csv
import pandas as pd
import os
from tqdm import tqdm


def open_csv_file(filename):
    """Opens a CSV file and returns a list of rows, where each row is a list of values."""
    try:
        with open(filename, "r") as f:
            return list(csv.reader(f))
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        raise
    except Exception as e:
        print(f"Error reading file '{filename}': {str(e)}")
        raise


def split_rows(rows, batch_row=7):
    """
    Splits a list of rows into batches of 7 rows each.
    This function handles the entire dataset, not just the first 7 rows.
    """
    total_batches = len(rows) // batch_row
    return [rows[i:i + batch_row] for i in range(0, total_batches * batch_row, batch_row)]


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
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df = pd.DataFrame(data)

        if df.empty:
            print(f"Warning: Empty data for file {filename}")
            return

        if file_type == 'csv':
            df.to_csv(filename, index=False, header=False)
        elif file_type == 'parquet':
            df.to_parquet(filename, index=False)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        print(f"Error saving file '{filename}': {str(e)}")
        raise


def process_data(folder, filename, output_files, file_formats, processed_data_dir):
    """Process data with error handling and progress tracking."""
    try:
        full_path = os.path.join(folder, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")

        # Read CSV file
        print(f"Reading file: {full_path}")
        raw_rows = open_csv_file(full_path)
        print(f"Total rows read: {len(raw_rows)}")

        # Split into batches of 7 rows
        data_points = split_rows(raw_rows)
        print(f"Number of batches: {len(data_points)}")

        if not data_points:
            raise ValueError("No data points found in the file")

        # Create progress bar
        total_operations = len(output_files) * len(file_formats)
        with tqdm(total=total_operations, desc="Processing data points") as pbar:
            for name, row_index in output_files.items():
                try:
                    print(f"\nProcessing {name} (row index {row_index})...")
                    data = extract_values(data_points, row_index)
                    print(f"Processed {len(data)} batches for {name}")

                    for file_format in file_formats:
                        output_file = os.path.join(processed_data_dir, f"{name}.{file_format}")
                        save_data(output_file, data, file_format)
                        print(f"\nSaved {output_file}")
                        pbar.update(1)
                except Exception as e:
                    print(f"Error processing {name} at row index {row_index}: {str(e)}")
                    raise

    except Exception as e:
        print(f"Error during data processing: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        print("Select the folder to process:")
        print("1 - Full Data")
        print("2 - Cyclic Data")
        print("3 - Monotonic Data")

        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == '1':
            FOLDER = "E:\OriginalData\Run_Full2/FullData"
            FILENAME = "Full_Data.csv"
            PROCESSED_DATA_DIR = f"RCWall_Data/Run_Full2/FullData"
        elif choice == '2':
            FOLDER = "RE:\OriginalData\Run_Full2/CyclicData"
            FILENAME = "Cyclic_Data.csv"
            PROCESSED_DATA_DIR = f"RCWall_Data/Run_Full2/CyclicData"
        elif choice == '3':
            FOLDER = "E:\OriginalData\Run_Full2/MonotonicData"
            FILENAME = "Monotonic_Data.csv"
            PROCESSED_DATA_DIR = f"RCWall_Data/Run_Full2/MonotonicData"
        else:
            raise ValueError("Invalid choice. Please choose 1, 2, or 3.")

        FILE_FORMATS = ['parquet']  # ['csv', 'parquet']
        OUTPUT_FILES = {
            'InputParameters': 0,
            'InputDisplacement': 1,
            'OutputShear': 2,
            'c1': 3,
            'a1': 4,
            'c2': 5,
            'a2': 6
        }

        process_data(FOLDER, FILENAME, OUTPUT_FILES, FILE_FORMATS, PROCESSED_DATA_DIR)
        print("\nProcessing completed successfully!")

    except Exception as e:
        print(f"\nProgram terminated with error: {str(e)}")
        exit(1)