import csv

# Path to your large CSV file
file_path = r'K:\RCShearWall_V2\RCWall_Data\original\Run_Final_Full\FullData\Full_Data.csv'

with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    # Read rows and print only rows at indices 0, 6, and 12
    for i, row in enumerate(reader):
        if i in [0, 7, 14]:  # Check if the row index is 0, 6, or 12
            print(row)