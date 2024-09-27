import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read data from CSV file
file_path = 'AE_metrics(04-2).csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

# Extract relevant information
optimizers = set(df['Optimizer'])
learning_rates = set(df['Learning Rate'])
batch_sizes = set(df['Batch Size'])

# Plotting
padding = 0.5  # Adjust this value to control the padding between bars
# Set consistent font settings
plt.rc('font', family='Times New Roman', size=10)  # Set default font for all elements
# Create subplots for each optimizer
fig, ax = plt.subplots(1, 3, figsize=(10, 10 / 3), sharey=True)
# Define a list of colors for the curves
colors = ['blue', 'red', 'green', 'black']

for optimizer in optimizers:
    fig, ax = plt.subplots(figsize=(12, 8))

    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            subset = df[(df['Optimizer'] == optimizer) & (df['Learning Rate'] == learning_rate) & (df['Batch Size'] == batch_size)]

            x_values = subset['Batch Size']
            y_values = subset['Training Loss']
            error_boxes = subset[['Training Loss', 'Validation Loss', 'Test Loss']].values.T  # Use .values to get a NumPy array

            for i, loss_type in enumerate(['Training Loss', 'Validation Loss', 'Test Loss']):
                # Adjust x-coordinate to add padding
                x_shifted = x_values + i * (padding + 0.02)
                ax.errorbar(x_shifted, subset[loss_type], yerr=error_boxes[i], label=f'LR={learning_rate}, {loss_type}', fmt='-o')

    # Set labels and legend
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss vs. Batch Size for {optimizer} Optimizer')
    ax.legend()

    # Show the plot
    plt.show()