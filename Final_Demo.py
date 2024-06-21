import tkinter as tk
from tkinter import Label, Button, Entry
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

save_dir = "/data/134-1/datasets/final_dataset"
annealing_accuracy = np.load('/data/134-1/datasets/final_dataset/annealing_accuracy.npy', allow_pickle=True)


def select_layer():
    root = tk.Tk()
    root.title("Select Layer")

    layer_var = tk.IntVar(root)
    layer_var.set(0)

    def submit():
        root.quit()
        root.destroy()

    Label(root, text="Layer").grid(row=1, column=0)
    Entry(root, textvariable=layer_var).grid(row=1, column=1)

    Button(root, text="Submit", command=submit).grid(row=2, columnspan=2)

    root.mainloop()
    return layer_var.get()

"""
def group_by_derivatives(images, params, indices):
    grouped = {}
    for img, par, idx in zip(images, params, indices):
        derivative_level = par[-4] + par[-3]
        if derivative_level not in grouped:
            grouped[derivative_level] = {'images': [], 'params': [], 'indices': []}
        grouped[derivative_level]['images'].append(img)
        grouped[derivative_level]['params'].append(par)
        grouped[derivative_level]['indices'].append(idx)
    return grouped

"""
def group_by_derivatives(images, params, indices):
    grouped = {}
    for img, par, idx in zip(images, params, indices):
        derivative_level = par[-4] + par[-3]
        if derivative_level>0:
            derivative_level = 1
        if derivative_level not in grouped:
            grouped[derivative_level] = {'images': [], 'params': [], 'indices': []}
        grouped[derivative_level]['images'].append(img)
        grouped[derivative_level]['params'].append(par)
        grouped[derivative_level]['indices'].append(idx)
    return grouped


def plot_grouped_images(axs, NF_matrix, layer, start_row, grouped, title, nf_only=False):
    row = start_row
    col = 0
    max_cols = 0
    for derivative_level, data in sorted(grouped.items()):
        if len(data['images']) == 0:
            continue
        axs[row, 2].set_title(f"{title} Derivative {derivative_level}", fontsize=8)
        axs[row, 0].axis('off')
        col = 1
        for img, par, idx in zip(data['images'], data['params'], data['indices']):
            if nf_only and sum(par[-1]) != 765:
                continue  # Skip non-NF images if nf_only is True
            if col >= axs.shape[1]:
                row += 1
                col = 0
            img_array = np.array(img)
            nf_image = NF_matrix[layer][idx]
            img_array = img_array.astype(float) / 255.0
            combined_image = np.concatenate((img_array, nf_image),
                                            axis=1 if img_array.shape[0] == nf_image.shape[0] else 0)
            axs[row, col].imshow(nf_image, cmap='gray')
            axs[row, col].axis('off')
            col += 1
        row += 1
        max_cols = max(max_cols, col)
    return row, max_cols

def main():
    global save_dir
    best_images_matrix = np.load(os.path.join(save_dir, 'best_images_matrix.npy'), allow_pickle=True)
    NF_matrix = np.load(os.path.join(save_dir, 'NF_matrix.npy'), allow_pickle=True)
    layer = select_layer()
    best_images_layer = best_images_matrix[layer]

    if layer < best_images_matrix.shape[0]:
        images = []
        params = []
        original_indices = []  # To keep track of original neuron indices
        not_fitted_images = {}  # To store images with low accuracy
        single_color_images = []
        for idx, neuron_data in enumerate(best_images_layer):
            if neuron_data is not None and neuron_data[0] is not None:
                if annealing_accuracy[layer][idx] > 0.70:

                    if np.all(np.std(np.array(NF_matrix[layer][idx]), axis=(0, 1)) < 0.075):
                        single_color_images.append(NF_matrix[layer][idx])

                    else:
                        images.append(neuron_data[0])
                        params.append(neuron_data[1])
                        original_indices.append(idx)  # Store the original index
                else:
                    not_fitted_images[idx] = neuron_data[0]

        num_images = len(images)
        print(num_images+len(single_color_images))
        sigmoid_images = []
        gaussian_images = []
        sigmoid_params = []
        gaussian_params = []
        sigmoid_indices = []
        gaussian_indices = []

        for idx_par, par in enumerate(params):
            if len(par) == 9:
                gaussian_images.append(images[idx_par])
                gaussian_params.append(par)
                gaussian_indices.append(original_indices[idx_par])
            else:
                sigmoid_images.append(images[idx_par])
                sigmoid_params.append(par)
                sigmoid_indices.append(original_indices[idx_par])

        if num_images > 0:
            # Sort images by angle
            sigmoid_sorted_indices = sorted(range(len(sigmoid_params)), key=lambda i: sigmoid_params[i][2])
            gaussian_sorted_indices = sorted(range(len(gaussian_params)), key=lambda i: gaussian_params[i][4])

            sigmoid_images = [sigmoid_images[i] for i in sigmoid_sorted_indices]
            gaussian_images = [gaussian_images[i] for i in gaussian_sorted_indices]

            sigmoid_params = [sigmoid_params[i] for i in sigmoid_sorted_indices]
            gaussian_params = [gaussian_params[i] for i in gaussian_sorted_indices]

            sorted_sigmoid_indices = [sigmoid_indices[i] for i in sigmoid_sorted_indices]
            sorted_gaussian_indices = [gaussian_indices[i] for i in gaussian_sorted_indices]

            # Separate BW and colored images
            sigmoid_bw_images = [img for img, par in zip(sigmoid_images, sigmoid_params) if sum(par[-1]) == 765]
            sigmoid_colored_images = [img for img, par in zip(sigmoid_images, sigmoid_params) if sum(par[-1]) != 765]
            sigmoid_bw_params = [par for par in sigmoid_params if sum(par[-1]) == 765]
            sigmoid_colored_params = [par for par in sigmoid_params if sum(par[-1]) != 765]
            sorted_sigmoid_bw_indices = [sorted_sigmoid_indices[i] for i, par in enumerate(sigmoid_params) if
                                         sum(par[-1]) == 765]
            sorted_sigmoid_colored_indices = [sorted_sigmoid_indices[i] for i, par in enumerate(sigmoid_params) if
                                              sum(par[-1]) != 765]

            gaussian_bw_images = [img for img, par in zip(gaussian_images, gaussian_params) if sum(par[-1]) == 765]
            gaussian_colored_images = [img for img, par in zip(gaussian_images, gaussian_params) if sum(par[-1]) != 765]
            gaussian_bw_params = [par for par in gaussian_params if sum(par[-1]) == 765]
            gaussian_colored_params = [par for par in gaussian_params if sum(par[-1]) != 765]
            sorted_gaussian_bw_indices = [sorted_gaussian_indices[i] for i, par in enumerate(gaussian_params) if
                                          sum(par[-1]) == 765]
            sorted_gaussian_colored_indices = [sorted_gaussian_indices[i] for i, par in enumerate(gaussian_params) if
                                               sum(par[-1]) != 765]

            # Group by derivative levels
            sigmoid_bw_grouped = group_by_derivatives(sigmoid_bw_images, sigmoid_bw_params, sorted_sigmoid_bw_indices)
            sigmoid_colored_grouped = group_by_derivatives(sigmoid_colored_images, sigmoid_colored_params,
                                                           sorted_sigmoid_colored_indices)
            gaussian_bw_grouped = group_by_derivatives(gaussian_bw_images, gaussian_bw_params,
                                                       sorted_gaussian_bw_indices)
            gaussian_colored_grouped = group_by_derivatives(gaussian_colored_images, gaussian_colored_params,
                                                            sorted_gaussian_colored_indices)

            # Initialize figure and axes
            num_rows = (len(sigmoid_bw_grouped) + len(sigmoid_colored_grouped) +
                        len(gaussian_bw_grouped) + len(gaussian_colored_grouped) +1)
            num_cols = max(len(sigmoid_bw_images), len(sigmoid_colored_images),
                           len(gaussian_bw_images), len(gaussian_colored_images)+1)
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows), constrained_layout=True,
                                    gridspec_kw={'hspace': 0.1})

            fig.suptitle(f"Layer: {layer}", fontsize=16)



            # Plot images for each group
            current_row = 0
            if sigmoid_bw_grouped:
                current_row, max_cols = plot_grouped_images(axs, NF_matrix, layer, current_row, sigmoid_bw_grouped, "Sigmoid BW")
            if sigmoid_colored_grouped:
                current_row, max_cols = plot_grouped_images(axs, NF_matrix, layer, current_row, sigmoid_colored_grouped,
                                                            "Sigmoid Colored")
            if gaussian_bw_grouped:
                current_row, max_cols = plot_grouped_images(axs, NF_matrix, layer, current_row, gaussian_bw_grouped, "Gaussian BW")
            if gaussian_colored_grouped:
                current_row, max_cols = plot_grouped_images(axs, NF_matrix, layer, current_row, gaussian_colored_grouped, "Gaussian Colored")

            if single_color_images:
                axs[num_rows-1, 2].set_title(f"Single Color Images", fontsize=8)
                for i, image in enumerate(single_color_images):
                    axs[num_rows-1, i+1].imshow(image)
                    axs[num_rows-1, i+1].axis('off')

            # Hide any unused subplots
            for ax in axs.flatten():
                if not ax.has_data():
                    ax.axis('off')

            plt.show()

        # Plot not fitted images
        if not_fitted_images:
            num_not_fitted = len(not_fitted_images)
            num_cols = 3  # Number of columns per row
            num_rows = (num_not_fitted + num_cols - 1) // num_cols  # Calculate number of rows needed

            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5), constrained_layout=True)
            fig.suptitle(f"Not Fitted Images for Layer: {layer}", fontsize=16)

        # Plot not fitted images
        if not_fitted_images:
            num_not_fitted = len(not_fitted_images)
            num_cols = int(np.ceil(np.sqrt(num_not_fitted)))  # Number of columns
            num_rows = int(np.ceil(num_not_fitted / num_cols))  # Number of rows

            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15), constrained_layout=True)
            fig.suptitle(f"Not Fitted Images for Layer: {layer}", fontsize=14)

            for idx, (key, img) in enumerate(not_fitted_images.items()):
                row = idx // num_cols
                col = idx % num_cols

                img_array = np.array(img)
                img_array = img_array.astype(float) / 255.0

                nf_image = NF_matrix[layer][key]
                combined_image = np.concatenate((img_array, nf_image),
                                                axis=1 if img_array.shape[0] == nf_image.shape[0] else 0)

                axs[row, col].imshow(combined_image, cmap='gray')
                axs[row, col].set_title(f"Neuron {key}", fontsize=8)
                axs[row, col].axis('off')

            # Hide any unused subplots
            """
            for ax in axs.flatten():
                if not ax.has_data():
                    ax.axis('off')
            """
            plt.show()

        else:
            print("No not fitted images found for the specified layer.")


if __name__ == "__main__":
    main()
