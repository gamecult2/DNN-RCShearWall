import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

# Read the data into a pandas DataFrame (replace 'your_data.csv' with your actual file)
df = pd.read_csv('AE_metrics(04-2).csv', delimiter=',')
df = pd.read_csv('Bi-LSTM_metrics(04-2).csv', delimiter=',')
df = pd.read_csv('CNN_metrics(04-2).csv', delimiter=',')

# Extract unique optimizers from the 'Optimizer' column
optimizers = df['Optimizer'].unique()

# Create an empty dictionary to store tensors for each optimizer
optimizer_tensors = {}

# Iterate over each optimizer
for optimizer in optimizers:
    # Filter DataFrame for the current optimizer
    optimizer_df = df[df['Optimizer'] == optimizer]

    # Extract unique learning rates and batch sizes
    learning_rates = optimizer_df['Learning Rate'].unique()
    batch_sizes = optimizer_df['Batch Size'].unique()

    # Create a 3D tensor for each optimizer
    tensor = np.zeros((len(learning_rates), len(batch_sizes), 3))

    # Iterate over each combination of learning rate and batch size
    for i, lr in enumerate(learning_rates):
        for j, batch_size in enumerate(batch_sizes):
            # Extract loss values for the current combination
            losses = optimizer_df[(optimizer_df['Learning Rate'] == lr) & (optimizer_df['Batch Size'] == batch_size)][['Training Loss', 'Validation Loss', 'Test Loss']].values[0]

            # Assign loss values to the tensor
            tensor[i, j, :] = losses

    # Add the tensor to the dictionary
    optimizer_tensors[optimizer] = tensor

# Access the tensor for a specific optimizer
adam_tensor = optimizer_tensors['Adam']
sgd_tensor = optimizer_tensors['SGD']
rmsprop_tensor = optimizer_tensors['RMSprop']

# Define the grid
lr_values = [0.00001, 0.0001, 0.001, 0.01]
batch_size_values = [8, 16, 32, 64]

# Create a meshgrid for LR and Batch Size
LR, Batch_Size = np.meshgrid(batch_size_values, lr_values)

# Set consistent font settings
plt.rc('font', family='Times New Roman', size=10)  # Set default font for all elements
# Create subplots for each optimizer
fig, axs = plt.subplots(1, 3, figsize=(9.5, 8 / 3), sharey=True, gridspec_kw={'top': 0.777, 'bottom': 0.184, 'left': 0.052, 'right': 0.994, 'hspace': 0.22, 'wspace': 0.109})

# Set the default font family
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
colors = ['blue', 'red', 'green', 'black']

# Iterate over optimizers
for i, (opt_tensor, opt_name) in enumerate(zip([adam_tensor, sgd_tensor, rmsprop_tensor], ['Adam', 'SGD', 'RMSprop'])):
    axs[i].set_title(f'Optimizer: {opt_name}', pad=28)

    for k, lr in enumerate(lr_values):
        axs[i].plot(batch_size_values, opt_tensor[k, :, 2].T * 1000, marker='x', label=f'(LR={lr})', color=colors[k], linewidth=0.4)

    axs[i].set_xlabel('Batch Size')
    axs[i].set_ylabel('Loss')
    axs[i].set_xticks(batch_size_values)  # Set X-axis ticks
    # axs[i].set_ylim(-6e-05 * 1000)  # Set Y-axis limits
    axs[i].set_ylim(-6e-05 * 1000, 0.00075 * 1000)  # Set Y-axis limits
    axs[i].grid(linestyle='dotted')
    # axs[i].legend(fontsize='small')  # Adjust legend position
    # axs[i].legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), ncol=len(lr_values)/2, frameon=True, fontsize='small')  # Adjust legend position

# Add the x10^3 text only to the first graph
axs[0].text(-0.08, 1.0, r'$10^3$', transform=axs[0].transAxes, fontsize=10, fontfamily='Times New Roman')  # Add the floating text

# Create a common legend at the top center
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=len(lr_values), frameon=True, labels=[f'(LR={lr})' for opt_name in ['Adam'] for lr in lr_values])

plt.tight_layout()
plt.savefig('CNN_metrics(04-2).svg', format='svg', dpi=300, bbox_inches='tight')
plt.show()
