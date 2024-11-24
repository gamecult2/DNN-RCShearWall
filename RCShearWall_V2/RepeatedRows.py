import pandas as pd
import sys
import os
from datetime import datetime
import csv
from pathlib import Path


def search_csv_files(folder_path, search_value):
    """
    Search through all CSV files in a folder for a specific value.
    Returns a list of tuples containing (filename, row numbers) where value was found.
    """
    results = []

    # Get all CSV files in the folder
    csv_files = list(Path(folder_path).glob('*.csv'))
    total_files = len(csv_files)

    print(f"\nSearching through {total_files} CSV files in {folder_path}")

    for file_num, file_path in enumerate(csv_files, 1):
        try:
            print(f"\rProcessing file {file_num}/{total_files}: {file_path.name}", end="")

            # Read the CSV file in chunks to handle large files
            chunk_size = 10000
            found_rows = []

            # First detect the delimiter
            with open(file_path, 'r') as f:
                first_line = f.readline()
                try:
                    dialect = csv.Sniffer().sniff(first_line)
                    delimiter = dialect.delimiter
                except:
                    delimiter = ','  # Default to comma if detection fails

            # Read and search the file in chunks
            for chunk in pd.read_csv(file_path,
                                     header=None,
                                     delimiter=delimiter,
                                     chunksize=chunk_size,
                                     on_bad_lines='skip'):
                # Search for the value in the entire chunk
                mask = chunk.astype(str).apply(lambda x: x.str.contains(str(search_value), na=False))
                if mask.any().any():
                    # Get the row numbers where value was found
                    chunk_rows = mask.any(axis=1)
                    found_rows.extend(chunk_rows[chunk_rows].index.tolist())

            if found_rows:
                results.append((str(file_path), found_rows))

        except Exception as e:
            print(f"\nError processing {file_path.name}: {str(e)}")
            continue

    print("\n\nSearch completed!")
    return results


def main():
    # Get folder path and search value from user
    folder_path = input("Enter the folder path containing CSV files: ")
    search_value = input("Enter the value to search for: ")

    # Search for the value in all CSV files
    results = search_csv_files(folder_path, search_value)

    if not results:
        print("No files found containing the specified value.")
        return

    print("\nFiles containing the specified value:")
    for i, (file_path, rows) in enumerate(results, 1):
        print(f"\n{i}. {os.path.basename(file_path)}")
        print(f"   Found in rows: {', '.join(str(r + 1) for r in rows[:5])}", end="")
        if len(rows) > 5:
            print(f" and {len(rows) - 5} more...")
        else:
            print()

if __name__ == "__main__":
    main()

    # CyclicData   613936
    # K:\RCShearWall_V2\RCWall_Data\Run_1\FullData    Done
    # K:\RCShearWall_V2\RCWall_Data\Run_1\CyclicData
    # K:\RCShearWall_V2\RCWall_Data\Run_2\FullData
    # K:\RCShearWall_V2\RCWall_Data\Run_2\CyclicData
