import csv
from tqdm import tqdm
from collections import Counter


def open_csv_file(filename):
    """Opens a CSV file and returns a list of rows, where each row is a list of values."""
    with open(filename, "r") as f:
        return list(csv.reader(f))


def remove_lines(rows):
    """Removes lines where the third line has a value greater than 27000, along with the previous two lines. Also removes the first two lines if the third line is repeated."""
    lines_to_remove = []
    third_line_values = [tuple(row) for row in rows[2::3]]
    third_line_hashes = [hash(str(row)) for row in third_line_values]
    repeated_hashes = [h for h, count in Counter(third_line_hashes).items() if count > 1]

    for i in tqdm(range(2, len(rows), 3), desc="Removing lines", unit="line"):
        current_row = tuple(rows[i])
        current_hash = hash(str(current_row))
        if any(abs(float(value)) > 27000 for value in rows[i]) or current_hash in repeated_hashes:
            lines_to_remove.extend([i - 2, i - 1, i])
        elif current_hash in repeated_hashes:
            lines_to_remove.extend([i - 2, i - 1])

    return [row for i, row in enumerate(rows) if i not in lines_to_remove], len(lines_to_remove)


def save_data(filename, data):
    """Saves data to a CSV file."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)


if __name__ == "__main__":
    # Configuration
    FOLDER = "RCWall_Data"
    FILENAME = "Original_Data/Original_Data.csv"
    OUTPUT_FILENAME = "Original_Data/Filtered_Data.csv"

    # Open CSV file
    rows = open_csv_file(f"{FOLDER}/{FILENAME}")

    # Remove lines with values greater than 27000 and repeated third lines, including the first two lines
    filtered_rows, num_lines_cleaned = remove_lines(rows)

    # Save the filtered data to a new CSV file
    save_data(f"{FOLDER}/{OUTPUT_FILENAME}", filtered_rows)

    print(f"Processing completed successfully! {num_lines_cleaned} lines cleaned.")
