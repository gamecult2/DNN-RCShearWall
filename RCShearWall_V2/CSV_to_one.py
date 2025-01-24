import csv
import os

# Ask the user to select the folder to process
print("Select the folder to process:")
print("1 - Full Data")
print("2 - Cyclic Data")
print("3 - Monotonic Data")
choice = input("Enter your choice (1, 2, or 3): ").strip()

# Define folder and output file paths based on the user's choice  ---   Run_Full  -----    Run_1    -----   Run_2
if choice == "1":
    folder = r"K:\RCShearWall_V2\RCWall_Data\Run_Final_Full\FullData"
    output_file = r"K:\RCShearWall_V2\RCWall_Data\original\Run_Final_Full\FullData\Full_Data.csv"
elif choice == "2":
    folder = r"K:\RCShearWall_V2\RCWall_Data\Run_Final_Full\CyclicData"
    output_file = r"K:\RCShearWall_V2\RCWall_Data\original\Run_Final_Full\CyclicData\Cyclic_Data.csv"
elif choice == "3":
    folder = r"K:\RCShearWall_V2\RCWall_Data\Run_Final_Full\MonotonicData"
    output_file = r"K:\RCShearWall_V2\RCWall_Data\original\Run_Final_Full\MonotonicData\Monotonic_Data.csv"
else:
    print("Invalid choice. Exiting the program.")
    exit()

# Check if the output file already exists
if os.path.exists(output_file):
    print(f"Output file already exists: {output_file}")
    print("Operation aborted to avoid overwriting existing data.")
    exit()

# Ensure the output folder exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Iterate through files in the directory
all_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
p = 0
# Open output file in append mode (to avoid overwriting existing data)
with open(output_file, 'a', newline='') as combined_file:
    writer = csv.writer(combined_file)

    # Skip header row if existing files already have headers
    skip_header = True

    # Loop through each CSV file
    for filename in all_files:
        file_path = os.path.join(folder, filename)

        # Open each file for reading
        with open(file_path, 'r', newline='') as individual_file:
            reader = csv.reader(individual_file)

            # Write rows from each file to the combined file
            writer.writerows(reader)
            p += 1
            print(f"File done {p}")

print(f"CSV files concatenated successfully. Output: {output_file}")
