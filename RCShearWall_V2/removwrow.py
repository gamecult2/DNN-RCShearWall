import pandas as pd


def remove_rows_from_csv(csv_file, rows_to_remove):
    """Removes specific rows from a CSV file.

    Args:
        csv_file: The path to the CSV file.
        rows_to_remove: A list of indices of the rows to remove (0-based).

    Returns:
        None
    """

    # Read the CSV file without headers
    df = pd.read_csv(csv_file, header=None)

    # Drop the specified rows
    df = df.drop(rows_to_remove, axis=0)

    # Reset the index of the DataFrame
    df = df.reset_index(drop=True)

    # Save the DataFrame back to the CSV file
    df.to_csv(csv_file, index=False, header=False)


def remove_rows_from_parquet(parquet_file, rows_to_remove):
    """Removes specific rows from a Parquet file.

    Args:
        parquet_file: The path to the Parquet file.
        rows_to_remove: A list of indices of the rows to remove (0-based).

    Returns:
        None
    """

    # Read the Parquet file
    df = pd.read_parquet(parquet_file)

    # Drop the specified rows
    df = df.drop(rows_to_remove, axis=0)

    # Reset the index of the DataFrame
    df = df.reset_index(drop=True)

    # Save the DataFrame back to the Parquet file
    df.to_parquet(parquet_file, index=False)


# Example usage:
parquet_file1 = 'RCWall_Data/Processed_Data/Data_30K/OutputCyclicShear.csv'
parquet_file2 = 'RCWall_Data/Processed_Data/Data_30K/InputCyclicDisplacement.csv'
parquet_file3 = 'RCWall_Data/Processed_Data/Data_30K/InputParameters.csv'

rows_to_remove = []  # Indices of the rows to remove (0-based)

remove_rows_from_csv(parquet_file1, rows_to_remove)
remove_rows_from_csv(parquet_file2, rows_to_remove)
remove_rows_from_csv(parquet_file3, rows_to_remove)


