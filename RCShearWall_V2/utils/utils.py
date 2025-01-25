import math
import numpy as np
import matplotlib.pyplot as plt
import os


# Plotting
def plot_metric(train_data, val_data, best_epoch, test_data, ylabel, title, model_name=None, save_fig=False):
    plt.figure(figsize=(6, 5))
    epochs = range(1, len(train_data) + 1)

    plt.plot(epochs, train_data, color='#3152a1', label=f"Training {ylabel}", linewidth=2)
    plt.plot(epochs, val_data, color='red', label=f"Validation {ylabel}", linewidth=2)
    if best_epoch:
        plt.scatter(best_epoch, val_data[best_epoch - 1], color='red', s=100, label="Best Model")
    # Add test loss as a triangle at the last epoch
    if test_data is not None:
        plt.scatter(len(epochs), test_data, color='green', marker='^', s=100, label=f"Test {ylabel}")

    plt.xlabel("Epochs", fontname='Times New Roman', fontsize=14)
    plt.ylabel(ylabel, fontname='Times New Roman', fontsize=14)
    plt.yticks(fontname='Times New Roman', fontsize=12)
    plt.xticks(fontname='Times New Roman', fontsize=12)
    plt.title(f"{title} Over Epochs", fontname='Times New Roman', fontsize=12)
    plt.legend(prop={'family': 'Times New Roman', 'size': 12})
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Save figure if requested
    if save_fig and model_name:
        # Create a directory to save the figures if it doesn't exist
        os.makedirs("training_metrics", exist_ok=True)
        # Save the figure as an SVG file
        filename = f"training_metrics/{model_name}_{title.replace(' ', '_')}.svg"
        plt.savefig(filename, format='svg', bbox_inches='tight')  # Save as SVG
    # plt.show()


# Function to save plots
def save_plots(test_index, predicted_shear, real_shear, new_input_displacement, model_name, save_fig=True):
    parent_dir = "results"
    output_dir = os.path.join(parent_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(test_index):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

        # Time Series Plot
        ax1.plot(predicted_shear[i], label=f'Predicted Shear - {i + 1}')
        ax1.plot(real_shear[i], label=f'Real Shear - {i + 1}')
        ax1.set_xlabel('Time Step', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
        ax1.set_ylabel('Shear Load', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
        ax1.set_title('Predicted Shear Time Series', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
        ax1.tick_params(axis='both', labelsize=14, labelcolor='black', colors='black')
        ax1.legend()
        ax1.grid()

        # Hysteresis Loop Plot
        ax2.plot(new_input_displacement[i], predicted_shear[i], label=f'Predicted Loop - {i + 1}')
        ax2.plot(new_input_displacement[i], real_shear[i], label=f'Real Loop - {i + 1}')
        ax2.set_xlabel('Displacement', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
        ax2.set_ylabel('Shear Load', {'fontname': 'Cambria', 'fontstyle': 'italic', 'size': 14})
        ax2.set_title('Predicted Hysteresis', {'fontname': 'Cambria', 'fontstyle': 'normal', 'size': 16})
        ax2.tick_params(axis='both', labelsize=14, labelcolor='black', colors='black')
        ax2.legend()
        ax2.grid()

        plt.tight_layout()
        if save_fig:
            fig.savefig(os.path.join(output_dir, f'figure_{i + 1}.png'))
        # plt.show()
        plt.close(fig)  # Close the figure to free up memory


# Function to save training summary
def save_training_summary(model, file_name, hyperparameters, metrics, best_epoch, test_metrics, total_params):
    training_folder = "results"
    training_dir = os.path.join(training_folder)
    os.makedirs(training_dir, exist_ok=True)
    file_path = os.path.join(training_dir, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=== Model Training Summary ===\n\n")

        # Save hyperparameters
        f.write("=== Hyperparameters ===\n")
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Model Type: {type(model).__name__}\n\n")

        # Save training and validation metrics
        f.write("=== Training and Validation Metrics ===\n")
        f.write("| Epoch | Train Loss | Train R2 | Val Loss | Val R2  |\n")
        f.write("|-------|------------|----------|----------|---------|\n")
        for epoch in range(len(metrics["train_losses"])):
            f.write(f"| {epoch + 1:^5} | {metrics['train_losses'][epoch]:^10.4f} | {metrics['train_r2_scores'][epoch]:^8.4f} | {metrics['val_losses'][epoch]:^8.4f} | {metrics['val_r2_scores'][epoch]:^7.4f} |\n")
        f.write(f"\nBest Epoch: {best_epoch}\n\n")

        # Save test metrics
        f.write("=== Test Metrics ===\n")
        f.write(f"Test Loss: {test_metrics['test_loss']:.4f}\n")
        f.write(f"Test R2: {test_metrics['test_r2']:.4f}\n\n")

        # Save best validation metrics and corresponding training metrics
        best_epoch_index = best_epoch - 1
        f.write("=== Best Validation Metrics ===\n")
        f.write(f"Best Validation Loss: {metrics['val_losses'][best_epoch_index]:.4f}\n")
        f.write(f"Best Validation R²: {metrics['val_r2_scores'][best_epoch_index]:.4f}\n\n")

        f.write("=== Training Metrics at Best Validation Epoch ===\n")
        f.write(f"Training Loss: {metrics['train_losses'][best_epoch_index]:.4f}\n")
        f.write(f"Training R²: {metrics['train_r2_scores'][best_epoch_index]:.4f}\n\n")

        # Save model summary
        f.write("=== Model Summary ===\n")
        f.write(f"Total Parameters: {total_params:,}\n")

    print(f"Model training summary saved to '{file_path}'")
