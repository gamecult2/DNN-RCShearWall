import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# List of parameters (column names)
parameters = ['tw', 'hw', 'lw', 'lbe', 'fc', 'fyb', 'fyw', 'rouYb', 'rouYw', 'loadcoef']

# Read the CSV file without headers
df = pd.read_csv('RCWall_Data/Dataset_full/Parameters.csv', header=None, names=parameters)# , nrows=50000)

# Set up the plot
fig, axs = plt.subplots(2, 5, figsize=(20, 10))
fig.suptitle('Histograms of Design Parameters', fontsize=16)

# Plot histogram for each parameter
for i, param in enumerate(parameters):
    row = i // 5
    col = i % 5

    # Histogram
    sns.histplot(df[param], kde=True, ax=axs[row, col], color='skyblue', edgecolor='black')
    axs[row, col].set_title(param)
    axs[row, col].set_xlabel('')

    # Remove y-axis label to save space
    axs[row, col].set_ylabel('')

    # Add mean and median lines
    mean = df[param].mean()
    median = df[param].median()
    axs[row, col].axvline(mean, color='red', linestyle='dashed', linewidth=1, label='Mean')
    axs[row, col].axvline(median, color='green', linestyle='dashed', linewidth=1, label='Median')

    # Add legend
    if i == 0:  # Only add legend to the first subplot to avoid repetition
        axs[row, col].legend()

# Adjust layout and save the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('parameter_histograms.png')
plt.show()
plt.close()

# Calculate and print basic statistics for each parameter
for param in parameters:
    print(f"\nStatistics for {param}:")
    print(df[param].describe())

# Calculate and print correlation matrix
correlation_matrix = df[parameters].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap of Design Parameters')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
plt.close()

print("\nAnalysis complete. Histogram plot and correlation heatmap have been saved as PNG files.")