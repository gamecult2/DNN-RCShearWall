import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# read the data from a csv file
data = pd.read_csv("AE_metrics(04-2).csv")

# get the unique values of the optimizer and learning rate columns
optimizers = data["Optimizer"].unique()
learning_rates = data["Learning Rate"].unique()

# Calculate A4 page dimensions in inches
a4_width_inches = 9.5
a4_height_inches = 8


# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.rc('font', family='Times New Roman', size=10)  # Set default font for all elements
fig, axes = plt.subplots(1, 3, figsize=(a4_width_inches, a4_height_inches / 3), sharey=True) #, gridspec_kw={'top': 0.752, 'bottom': 0.184, 'left': 0.082, 'right': 0.99, 'hspace': 0.2, 'wspace': 0.12})
padding = 0.5
colors = ['blue', 'red', 'green', 'black']
# Plotting
for i, opt in enumerate(optimizers):
    # filter the data by the optimizer
    df = data[data["Optimizer"] == opt]
    # get the axis for the current optimizer
    ax = axes[i]
    # loop over the learning rates
    for j, lr in enumerate(learning_rates):
        # filter the data by the learning rate
        df_lr = df[df["Learning Rate"] == lr]
        # get the x, y, and error values
        x = df_lr["Batch Size"]
        y = df_lr["Training Loss"]
        y2 = df_lr["Test Loss"]
        upperlimits = np.abs(df_lr["Test Loss"].values-y)
        # upperlimits = df_lr["Test Loss"].values-y
        lowerlimits = np.abs(y-df_lr["Test Loss"].values)
        yerr = np.array([df_lr["Test Loss"]])
        # yerr = np.array([df_lr["Training Loss"]])
        # yerr = np.array([y-df_lr["Training Loss"].values, y-df_lr["Test Loss"].values])
        x_shifted = x + j * (padding + 0.02)
        # yerr = np.array([np.abs(df_lr["Training Loss"].values), np.abs(df_lr["Test Loss"].values)])
        # ax.errorbar(x_shifted, y, lowerlimits, uplims=True, label=f"Learning Rate = {lr}", capsize=3)

        # plot the data with error bars
        ax.errorbar(x_shifted, y, upperlimits, uplims=True, fmt="s-", label=f"LR = {lr}", capsize=3, linewidth=0.5)

        # ax.fill_between(x, y, y2, alpha=.5, linewidth=0, color=colors[j])
        # ax.plot(x, y, linewidth=2)
        # ax.plot(x, y2, linewidth=2)
        # ax.plot(x, (y1 + y2) / 2, linewidth=2)

    # set the axis labels and title
    ax.set_xticks(x)
    ax.set_ylim(-6e-05, 0.00055)  # Set Y-axis limits
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Loss")
    ax.set_title(f'Optimizer: {opt}', pad=0)
    ax.grid(linestyle='dotted')
    # show the legend
    ax.legend(fontsize='small')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), ncol=len(learning_rates)/2, frameon=True, fontsize='small')  # Adjust legend position

# save and show the figure
plt.savefig("plot_by_optimizer.png")
plt.show()