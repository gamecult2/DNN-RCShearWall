import os
import pandas as pd
import numpy as np

'''

def search_csv_files_for_value(folder_path, search_value, tolerance=0.01):
    matching_files = []
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    total_files = len(csv_files)

    print(f"Total CSV files to search: {total_files}")

    for index, filename in enumerate(csv_files, 1):
        file_path = os.path.join(folder_path, filename)
        print(f"Searching file {index}/{total_files}: {filename} ", end='')

        try:
            # Read the CSV file as float
            df = pd.read_csv(file_path, dtype=float)
            # Print the first 10 rows of the DataFrame

            # Flatten and handle NaN values
            flat_values = df.values.flatten()
            flat_values = flat_values[~np.isnan(flat_values)]

            # Check if the search value is within tolerance
            if np.any(np.abs(flat_values == search_value)):
                print("✓ VALUE FOUND!")
                matching_files.append(filename)
            else:
                print("- Value not found")

        except pd.errors.EmptyDataError:
            print("- Empty file")
        except Exception as e:
            print(f"- Error searching file: {e}")

    return matching_files


# Example usage
if __name__ == "__main__":
    # Get folder path from user
    folder_path = input("Enter the full path to the folder containing CSV files: ")

    # Get search value from user
    search_value = float(input("Enter the numeric value to search for: "))

    # Optional: Get tolerance (default 0.01)
    try:
        tolerance = float(input("Enter tolerance (default 0.01): ") or "0.01")
    except ValueError:
        tolerance = 0.01

    # Find CSV files containing the value
    result = search_csv_files_for_value(folder_path, search_value, tolerance)

    # Print final results
    print("\n--- Search Complete ---")
    if result:
        print("Files containing the value:")
        for file in result:
            print(file)
    else:
        print("No files found with the specified value.")


        # K:\RCShearWall_V2\RCWall_Data\OriginalData\Run_Full\FullData
        # K:\RCShearWall_V2\RCWall_Data\Run_2\CyclicData
        # -23356.9973
        # -26764.43, -26839.33, -26891.38, -26922.613, -26934.006
        # -23356.997
        # Check if the search value is within tolerance
        # -23356.997298705             5285.80899114798                   613936
        
        # 237	296	1476	2460	0.6	262	60	260	630	650	0.0458	0.0228	0.0027	0.0171	0.2374	613936	0

'''

import pandas as pd
import sys
import os
from datetime import datetime
import csv


def manage_csv_rows(input_file_path, start_row, end_row):
    try:
        # First, detect the number of columns and delimiter
        with open(input_file_path, 'r') as file:
            first_line = file.readline()
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(first_line)
            delimiter = dialect.delimiter

            # Count maximum number of columns
            file.seek(0)
            max_columns = max(len(row) for row in csv.reader(file, dialect))

        print(f"Detected delimiter: '{delimiter}'")
        print(f"Maximum number of columns: {max_columns}")

        # Read the CSV file with the detected parameters
        df = pd.read_csv(input_file_path,
                         header=None,
                         delimiter=delimiter,
                         on_bad_lines='skip',  # Skip lines with wrong number of fields
                         names=range(max_columns))  # Create column names from 0 to max_columns-1

        # Check if the rows exist
        if start_row > len(df) or end_row > len(df):
            print(f"Error: File only has {len(df)} rows. Cannot process rows {start_row} to {end_row}")
            return

        # Get the first 5 columns
        columns_to_show = list(range(min(17, max_columns)))

        # Print only the first 5 columns of the rows to be removed
        print("\nRows to be removed (showing first 5 columns):")
        print("--------------------------------------------")
        rows_to_remove = df.loc[start_row - 1:end_row - 1, columns_to_show]
        print(rows_to_remove.to_string(index=False))

        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name, file_extension = os.path.splitext(input_file_path)
        output_file_path = f"{file_name}_modified_{timestamp}{file_extension}"

        # Ask for confirmation
        while True:
            print(f"\nNew file will be saved as: {output_file_path}")
            confirm = input("Do you want to proceed with removing these rows? (yes/no): ").lower()
            if confirm in ['yes', 'no']:
                break
            print("Please enter 'yes' or 'no'")

        if confirm == 'yes':
            # Remove the rows (from the full DataFrame)
            new_df = pd.concat([df.iloc[:start_row - 1], df.iloc[end_row:]], ignore_index=True)

            # Save to new file without headers
            new_df.to_csv(output_file_path,
                          index=False,
                          header=False,
                          sep=delimiter,
                          na_rep='')  # Replace NaN with empty string

            print(f"\nRows {start_row} to {end_row} have been removed.")
            print(f"Modified data saved to: {output_file_path}")

            # Print row count information
            print(f"\nOriginal file row count: {len(df)}")
            print(f"New file row count: {len(new_df)}")
            print(f"Rows removed: {len(df) - len(new_df)}")
        else:
            print("\nOperation cancelled. No files were modified.")

    except FileNotFoundError:
        print(f"Error: File '{input_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Try specifying the delimiter manually if auto-detection fails.")


if __name__ == "__main__":
    # Example usage
    input_file_path = "RCWall_Data/OriginalData/Run_Full/FullData/Full_Data.csv"  # Replace with your CSV file path
    manage_csv_rows(input_file_path, 362783, 362789)
