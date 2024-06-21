import numpy as np
import matplotlib.pyplot as plt

# Assuming the necessary data is already defined
annealing_accuracy_2 = np.load('/data/134-1/datasets/final_dataset/annealing_accuracy.npy', allow_pickle=True)
best_images_matrix = np.load('/data/134-1/datasets/final_dataset/best_images_matrix.npy', allow_pickle=True)

num_layers = best_images_matrix.shape[0]
bw_annealing_acc = []
colored_annealing_acc = []
general_acc = []
num_layers = 6
# Iterate through each layer
for layer_idx in range(num_layers):
    best_images_layer = best_images_matrix[layer_idx]

    bw_acc_per_layer = []
    colored_acc_per_layer = []

    # Iterate through images and parameters in the current layer
    for idx_n, neuron_data in enumerate(best_images_layer):
        if neuron_data is not None and neuron_data[0] is not None and neuron_data[1] is not None:
            image = neuron_data[0]
            params = neuron_data[1]
            color_value = sum(params[-1])  # Assuming last parameter indicates color (255 for colored)

            if color_value == 765:
                bw_acc_per_layer.append(annealing_accuracy_2[layer_idx][idx_n])
            else:
                colored_acc_per_layer.append(annealing_accuracy_2[layer_idx][idx_n])


    # Calculate mean accuracy per layer for b&w and colored images
    mean_bw_acc = np.mean(bw_acc_per_layer) if bw_acc_per_layer else 0
    mean_colored_acc = np.mean(colored_acc_per_layer) if colored_acc_per_layer else 0
    bw_annealing_acc.append(mean_bw_acc)
    colored_annealing_acc.append(mean_colored_acc)
# Now bw_annealing_acc and colored_annealing_acc contain the mean accuracies for each layer
print(bw_annealing_acc)
print(colored_annealing_acc)
# Sample data (replace with your actual data)
num_layers = 6  # Example number of layers
layers = np.arange(num_layers)  # X-axis values (layers)
# Plotting parameters

# Adjust bar width and positions
bar_width = 0.2
opacity = 0.8
bar_positions_bw = layers - bar_width
bar_positions_colored = layers
bar_positions_general = layers + bar_width
magenta= '#9C1E1E'
plt.figure(figsize=(12, 8))

# Plot bars for black & white annealing accuracies
plt.bar(bar_positions_bw, bw_annealing_acc, bar_width, alpha=opacity, color='gray', label='Mean b&w Fit')

# Plot bars for colored annealing accuracies
plt.bar(bar_positions_colored, colored_annealing_acc, bar_width, alpha=opacity, color=magenta, label='Mean colored Fit')


# Plot a horizontal line for Top Scoring
plt.axhline(y=1, color='green', linestyle='--', label='Top Scoring', linewidth=2)

# Add value annotations on top of each bar (for bw and colored)
for xpos, ypos in zip(layers, bw_annealing_acc):
    plt.text(xpos - bar_width, ypos + 0.01, f'{ypos:.2f}', ha='center', va='bottom', color='gray')

for xpos, ypos in zip(layers, colored_annealing_acc):
    plt.text(xpos, ypos + 0.01, f'{ypos:.2f}', ha='center', va='bottom', color=magenta)


# Customize plot aesthetics
plt.xlabel('Layers', fontsize=12)
plt.ylabel('Mean Fit', fontsize=12)
plt.title('Mean Fit by Color/Layer', fontsize=14)
plt.xticks(layers, fontsize=10)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()
