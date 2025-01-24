import csv
import numpy as np
import os
import torch

# Define the input and output file paths
input_file = r"K:\RCShearWall_V2\RCWall_Data\original\Run_Final_Full\FullData\Full_Data.csv"
output_file = r"K:\RCShearWall_V2\RCWall_Data\original\Run_Final_Full\FullData\Full_Data_float16.csv"

# Set the chunk size
chunk_size = 40000  # Adjust this as necessary

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

# Get total file size for progress calculation
total_size = os.path.getsize(input_file)
processed_size = 0
chunk_count = 0

# Open the input and output files
with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    # Initialize a counter for row positions in batches
    row_count = 0

    # Process the file in chunks
    batch = []
    for row in reader:
        batch.append(row)
        processed_size += len(','.join(row).encode('utf-8')) + 1  # +1 for newline

        if len(batch) >= chunk_size:
            chunk_count += 1
            # Calculate progress percentage
            progress = (processed_size / total_size) * 100
            print(f"Processing chunk {chunk_count} - Progress: {progress:.2f}% ({processed_size:,} / {total_size:,} bytes)")

            # Process the current chunk (batch)
            for i, row in enumerate(batch):
                # Determine the row position and select the column range
                row_position = (row_count + i) % len(column_ranges)
                col_start, col_end = column_ranges[row_position]

                # Slice the row based on the column range
                row = row[col_start:col_end]

                # Convert the row elements to float32
                # row = [np.float16(val) for val in row]
                # Convert to bfloat16 using PyTorch
                row = [torch.tensor(float(val)).to(torch.bfloat16).item() for val in row]

                # Write the converted row to the output file
                writer.writerow(row)

            # Reset the batch
            batch = []

    # Write any remaining rows if they don't complete a full chunk
    if batch:
        chunk_count += 1
        print(f"Processing final chunk {chunk_count} - Progress: 100.00% (Remaining {len(batch)} rows)")

        for i, row in enumerate(batch):
            row_position = (row_count + i) % len(column_ranges)
            col_start, col_end = column_ranges[row_position]

            # Slice the row based on the column range
            row = row[col_start:col_end]

            # Convert to bfloat16 using PyTorch
            row = [torch.tensor(float(val)).to(torch.bfloat16).item() for val in row]

            # Write the converted row to the output file
            writer.writerow(row)

print(f"\nProcessing complete! Total chunks processed: {chunk_count}")