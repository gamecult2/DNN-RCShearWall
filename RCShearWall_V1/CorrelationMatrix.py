import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

# Assuming your data is in a CSV file
df = pd.read_csv("C:/Users/djerr/PycharmProjects/NN_Project/NeuralNetworkProject/RCWall_Data/cyclic(02)/Results Parameters Vs Shear.csv")

# Selecting relevant columns
input_params = ['tw', 'hw', 'lw', 'lbe', 'fc', 'fyb', 'fyw', 'rYb', 'rYw', 'load']
output_param = 'Vmax'

# Calculate sensitivity scores using Spearman correlation
sensitivity_scores = df[input_params].apply(lambda x: spearmanr(x, df[output_param]).correlation)

# Plotting the sensitivity chart with values inside the bars and coolwarm color
plt.rcParams.update({'font.size': 12, "font.family": ["Cambria", "Times New Roman"]})
plt.figure(figsize=(7, 3))
barplot = sns.barplot(x=sensitivity_scores.index, y=sensitivity_scores.values, palette="coolwarm", edgecolor=".2")

# Set color based on positive or negative values
cmap = plt.cm.coolwarm
norm = plt.Normalize(-1, 1)
for i, v in enumerate(sensitivity_scores):
    barplot.patches[i].set_facecolor(cmap(norm(v)))

# Display values inside the bars with two decimal places
for index, value in enumerate(sensitivity_scores.values):
    barplot.text(index, value, f"{value:.2f}", ha='center', va='bottom')

# Set y-axis limits between -1 and 1
plt.ylim(-1, 1)
plt.axhline(y=0, color='grey', linewidth=1)
# Add padding to the grid lines on the x-axis
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey', alpha=0.7, axis='y', markevery=0)
plt.xlabel('Input Parameters')
plt.ylabel(r"Sensitivity on Vmax"
           "\n" 
           "(Spearman Coefficient)")
plt.tight_layout()
plt.savefig("sensitivity_chart.svg")
plt.show()