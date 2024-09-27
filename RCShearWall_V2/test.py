import pandas as pd
import matplotlib.pyplot as plt

# Replace 'file_path' with the path to your CSV file
file_path = 'RCWall_Data/Processed_Data/Data_30K/OutputCyclicShear.csv'

# Replace 'row_index' with the index of the row you want to plot
# row_index = 87572   # 87574
row_index = 218303-1


# Read the CSV file
data = pd.read_csv(file_path, header=None)

# Check if the row_index is within the range of the DataFrame
if row_index < 0 or row_index >= len(data):
    print(f"Row index {row_index} is out of range.")
else:
    # Extract the specified row
    row = data.iloc[row_index]

    # Plot the row
    plt.figure(figsize=(10, 6))
    plt.plot(row.values, marker='o')
    plt.title(f'Row {row_index} of the CSV file')
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.xticks(rotation=90)  # Rotate x-axis labels if necessary
    plt.grid(True)
    plt.tight_layout()
    plt.show()
