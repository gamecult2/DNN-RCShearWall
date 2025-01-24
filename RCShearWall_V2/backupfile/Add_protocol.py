import csv
import pandas as pd


# Input files
displacement_file = "RCWall_Data/displacement_Data.csv"
parameters_file = "RCWall_Data/parameters_Data.csv"
output_file = "RCWall_Data/parameters_Data_with_analysis.csv"

# Step 1: Read the displacement data and determine the type of loading
analysis_types = []
print("Processing displacement data...")

with open(displacement_file, mode='r') as disp_file:
    reader = csv.reader(disp_file)
    row_count = 0
    for row in reader:
        row_count += 1
        # Convert row to floats
        values = list(map(float, row))
        # Check for cyclic or monotonic
        if any(v < 0 for v in values):  # Contains negative values
            analysis_types.append(0)  # Cyclic
            print(f"Row {row_count}: Detected cyclic loading (contains negative values).")
        else:
            analysis_types.append(1)  # Monotonic
            print(f"Row {row_count}: Detected monotonic loading (only positive values).")

print(f"Displacement data processed: {row_count} rows analyzed.")

# Calculate and print percentages
num_zeros = analysis_types.count(0)
num_ones = analysis_types.count(1)
total = len(analysis_types)
print(f"Percentage of cyclic (0): {num_zeros / total * 100:.2f}%")
print(f"Percentage of monotonic (1): {num_ones / total * 100:.2f}%")

# Step 2: Open parameters data and add the analysis type column
print("Processing parameters data and adding analysis type column...")

with open(parameters_file, mode='r') as param_file, \
        open(output_file, mode='w', newline='') as output_csv:
    reader = csv.reader(param_file)
    writer = csv.writer(output_csv)

    # Write rows with the added analysis type
    row_count = 0
    for row, analysis_type in zip(reader, analysis_types):
        row_count += 1
        writer.writerow(row + [analysis_type])
        print(f"Row {row_count}: Appended analysis type {analysis_type}.")

print(f"Updated file saved as {output_file}. Processed {row_count} rows.")

# ---------------------------------------------------------------------------

# Input files
displacement_file = "RCWall_Data/New_Data/InputCyclicDisplacement.parquet"
parameters_file = "RCWall_Data/New_Data/InputParameters.parquet"
output_file = "RCWall_Data/New_Data/InputCyclicShear.parquet"

# Step 1: Read the displacement data and determine the type of loading
print("Processing displacement data...")
displacement_df = pd.read_parquet(displacement_file)

# Determine loading type for each row
analysis_types = []
for index, row in displacement_df.iterrows():
    if any(value < 0 for value in row):  # Contains negative values
        analysis_types.append(0)  # Cyclic
        print(f"Row {index + 1}: Detected cyclic loading (contains negative values).")
    else:
        analysis_types.append(1)  # Monotonic
        print(f"Row {index + 1}: Detected monotonic loading (only positive values).")

print(f"Displacement data processed: {len(analysis_types)} rows analyzed.")

# Calculate and print percentages
num_zeros = analysis_types.count(0)
num_ones = analysis_types.count(1)
total = len(analysis_types)
print(f"Percentage of cyclic (0): {num_zeros / total * 100:.2f}%")
print(f"Percentage of monotonic (1): {num_ones / total * 100:.2f}%")

# Step 2: Add the analysis type to parameters data
print("Processing parameters data and adding analysis type column...")
parameters_df = pd.read_parquet(parameters_file)

# Add the new 'analysis_type' column
parameters_df['analysis_type'] = analysis_types

# Save the updated data to a new parquet file
parameters_df.to_parquet(output_file, index=False)
print(f"Updated file saved as {output_file}. Processed {len(parameters_df)} rows.")
