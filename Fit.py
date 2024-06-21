import numpy as np
import matplotlib.pyplot as plt

# Assuming the necessary data is already defined
grid_accuracy = np.load('/data/134-1/datasets/final_dataset/grid_accuracy.npy', allow_pickle=True)
annealing_accuracy = np.load('/data/134-1/datasets/final_dataset/annealing_accuracy.npy', allow_pickle=True)
best_images_matrix = np.load('/data/134-1/datasets/final_dataset/best_images_matrix.npy', allow_pickle=True)
neurons_per_layer = [64, 64, 128, 128, 256, 256]
num_layers = grid_accuracy.shape[0]

# Function to calculate mean accuracy per layer
def calculate_mean_accuracy_per_layer(accuracy_matrix, neurons_per_layer):
    mean_accuracies = []
    for layer_idx, neurons in enumerate(neurons_per_layer):
        valid_accuracies = accuracy_matrix[layer_idx][:neurons]
        mean_accuracy = np.mean(valid_accuracies)
        mean_accuracies.append(mean_accuracy)
        print("Layer", layer_idx, "Mean Accuracy:", mean_accuracy)
    return np.array(mean_accuracies)

# Calculate mean accuracies for both methods
mean_grid_accuracy_2 = calculate_mean_accuracy_per_layer(grid_accuracy, neurons_per_layer)
mean_annealing_accuracy_2 = calculate_mean_accuracy_per_layer(annealing_accuracy, neurons_per_layer)

# Create figure and subplots
plt.figure(figsize=(16, 12))
plt.suptitle('Fit for neurons in Layers', fontsize=16)

for layer_to_plot in range(6):
    num_neurons = neurons_per_layer[layer_to_plot]
    sorted_neurons_indices = np.argsort(annealing_accuracy[layer_to_plot][:num_neurons])[::-1]
    print(sorted_neurons_indices[-1])
    sorted_annealing_accuracy = annealing_accuracy[layer_to_plot][:num_neurons][sorted_neurons_indices]
    sorted_grid_accuracy = grid_accuracy[layer_to_plot][:num_neurons][sorted_neurons_indices]
    top_scoring_accuracy_neurons = [1] * num_neurons

    # Create subplots
    plt.subplot(3, 2, layer_to_plot + 1)  # 3 rows, 2 columns, subplot index starts from 1
    plt.scatter(range(len(sorted_neurons_indices)), sorted_grid_accuracy,
                label=f'Grid Search - Layer {layer_to_plot}', color='purple', s=50)
    plt.scatter(range(len(sorted_neurons_indices)), sorted_annealing_accuracy,
                label=f'Simulated Annealing - Layer {layer_to_plot}', color='orange', s=50)
    plt.plot(sorted_neurons_indices, top_scoring_accuracy_neurons, label='Top Scoring', color='green')

    # Customize subplot aesthetics
    plt.xlabel('Neurons sorted by Simulated Annealing Accuracy', fontsize=8)
    plt.ylabel('Fit', fontsize=8)
    plt.title(f'Layer {layer_to_plot}', fontsize=14)
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True)

# Adjust layout and padding
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Add some space at the top for the suptitle

# Show the plot
plt.show()
# Plot mean accuracy per layer for both methods
layers = [0,1,2,3,4,5]
# Create figure and set size
plt.figure(figsize=(10, 6))
# Plotting the bar plot
bar_width = 0.3
opacity = 0.8

# Positions for the bars
bar_positions_grid = np.arange(len(layers))
bar_positions_annealing = [pos + bar_width for pos in bar_positions_grid]

plt.figure(figsize=(10, 6))

# Plot bars for Grid Search
plt.bar(bar_positions_grid, mean_grid_accuracy_2, bar_width, alpha=opacity, color='purple', label='Grid Search')

# Plot bars for Simulated Annealing
plt.bar(bar_positions_annealing, mean_annealing_accuracy_2, bar_width, alpha=opacity, color='orange', label='Simulated Annealing')

# Plot a horizontal line for Top Scoring
plt.axhline(y=1, color='green', linestyle='--', label='Top Scoring')

# Customize plot aesthetics
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Mean Fit', fontsize=12)
plt.title('Mean Fit for each method/layer', fontsize=14)
plt.xticks(bar_positions_grid + bar_width / 2, layers)  # Use layers as x-axis labels
plt.legend(loc='upper right', fontsize=10)
plt.grid(True)

# Add value annotations on top of each bar
for xpos, ypos in zip(bar_positions_grid, mean_grid_accuracy_2):
    plt.text(xpos, ypos + 0.01, f'{ypos:.2f}', ha='center', va='bottom', color='purple')

for xpos, ypos in zip(bar_positions_annealing, mean_annealing_accuracy_2):
    plt.text(xpos, ypos + 0.01, f'{ypos:.2f}', ha='center', va='bottom', color='orange')

plt.tight_layout()
plt.show()
