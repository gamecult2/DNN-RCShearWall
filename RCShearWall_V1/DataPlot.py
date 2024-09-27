import pandas as pd
import matplotlib.pyplot as plt

# Define the file paths
file1 = "RCWall_Data/Dataset_cyclic/OutputCyclicDisplacement_values.csv"
file2 = "RCWall_Data/Dataset_cyclic/OutputCyclicShear_values.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Calculate the number of rows and columns for the subplots
num_rows = len(df1)  # You can also use len(df2) since both have the same number of rows
num_cols = 3  # The number of columns in each figure
plots_per_figure = 12

# Calculate the number of figures needed
num_figures = (num_rows - 1) // plots_per_figure + 1

# Iterate over the figures
for figure_num in range(num_figures):
    # Create a new figure for each batch of 12 plots
    fig, axes = plt.subplots(4, 3, figsize=(14, 10))

    # Calculate the range of rows for the current batch
    start_row = figure_num * plots_per_figure
    end_row = min((figure_num + 1) * plots_per_figure, num_rows)

    for i in range(start_row, end_row):
        # Extract the data from the current row of each DataFrame
        data1 = df1.iloc[i]
        data2 = df2.iloc[i]

        # Plot the data on the current subplot
        ax = axes[i % 4, i % 3]
        ax.plot(data1, data2, color="red", linestyle="-", linewidth=1.2, label='Output Displacement vs Shear Load')
        plt.axhline(0, color='black', linewidth=0.4)
        plt.axvline(0, color='black', linewidth=0.4)
        ax.set_xlabel(f"Displacement", {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 8})
        ax.set_ylabel(f"Shear", {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 8})
        ax.set_title(f"Sample {i + 1}", {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 10})

    # Adjust the subplot layout for the current figure
    fig.tight_layout()
    # Adjust the subplot layout for the current figure
    fig.subplots_adjust(left=0.050, right=0.985, bottom=0.070, top=0.970, hspace=0.3, wspace=0.280)

    # Show the current figure
    plt.show()
