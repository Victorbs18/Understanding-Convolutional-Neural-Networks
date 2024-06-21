"""
This file contains a toy examples to have a first contact with Nefesi, and Keras.
This file has been created with tensorflow (and tensorflow-gpu) 1.8.0, keras 2.2.0, and python 3.6 (with anaconda3 interpreter)
"""
import cv2
import torch
from torchvision import transforms
import numpy as np
import functools
from functions.network_data2 import NetworkData
import types
from functions.image import ImageDataset
import interface_DeepFramework.DeepFramework as DeepF
import torchvision.models as models
from torch import nn
import gc
import torchvision
from Retrain_Force_Selectivity.Vgg16_modifications_Normal import vgg16
from functions import read_activations
import math
import os
import clip
import torch
from sklearn.cluster import KMeans
from torchvision.datasets import CIFAR100
import matplotlib.pyplot as plt
# Load the model
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import pandas as pd
import colorsys
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, argrelextrema


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50', device)


def preproces_VGG16_3(imgs_hr):
    # img=np.array(imgs_hr)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tnsr = [preprocess(imgs_hr)]

    return tnsr


# Function to process images in batches

def get_parameters(image_name):
    if image_name[15] == '.':

        indices = (
            int(image_name[:3]), int(image_name[3:6]), int(image_name[6:9]), int(image_name[9:12]),
            int(image_name[12:15]))  # (a,b,theta,dx,dy) .3d

    else:

        indices = (
        int(image_name[:3]), int(image_name[3:6]), int(image_name[6:9]), int(image_name[9:12]),
        int(image_name[12:15]), int(image_name[15:18]),int(image_name[18:21])) # (mu_x,mu_y,sigma_x,sigma_y,theta,dx,dy) .3d

    return indices

def process_images_in_batches(image_folder,list_images):
    images = []

    for image_name in list_images:  # Sorting to ensure consistent order
            image = Image.open(os.path.join(image_folder, image_name)).convert("RGB")
            images.append(image)
    return images

def process_col_images_in_batches(image_folder,list_images,col1,col2):
    col_images = []

    for image_name in list_images:  # Sorting to ensure consistent order
            image = Image.open(os.path.join(image_folder, image_name)).convert("RGB")
            col_image = map_color(image,col1,col2)
            col_images.append(col_image)
    return col_images

def top_k_elements(lst, k):
    indexed_list = list(enumerate(lst))
    sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
    top_k = sorted_list[:k]
    activations = {}
    for item in top_k:
        index, value = item
        activations[index] = value
    return activations


def plot_layer_mosaic(layer_mosaic, list_images_layer, Nefesi_model, parameters_matrix,gaussian_parameters_matrix, annealing_params, layer, layernum, layer_idx,image_folder, best_images_matrix=None, grid_accuracy=None, annealing_accuracy=None, show_images=False, save_images=False):
    print(layer_mosaic.shape[0])
    for i in range(layer_mosaic.shape[1]):
        neuron = Nefesi_model.get_neuron_of_layer('features.' + layernum, i)
        col_sel = layer.neurons_data[i].selectivity_idx['color']

        patch = Image.fromarray((Nefesi_model.dataset.get_patch(img_name=neuron.images_id[0],
                                                   crop_pos=neuron.xy_locations[0],
                                                   K=layer.receptive_field_Kernel, P=layer.receptive_field_Padding,
                                                   S=layer.receptive_field_Stride, mode='constant') * 255).astype(
            np.uint8))
        orig_act = neuron.activations[0]
        print("Original activation neuron", i, ":", orig_act)
        activations = top_k_elements(layer_mosaic[:, i], 5)
        print(activations)
        keysList = list(activations.keys())
        print()
        if col_sel > 0.2:
            col_images = []
            top_colors = scan_topColors(patch, n=2)
            #print(top_colors)
            """"
            for ind_img, img in enumerate(list_images_layer):
                img = Image.open(os.path.join(image_folder, list_images_layer[ind_img])).convert("RGB")
                col_img = map_color(img,top_colors[0],top_colors[1])
                col_images.append(col_img)
                col_neuron_mosaic = create_patch_mosaic_for_activations(col_images, layer.receptive_field_Kernel, layer.receptive_field_Padding,layer.receptive_field_Stride, Nefesi_model.model, preproces_VGG16_3, layernum, [i])
            """
            batch_size = 10
            num_images = layer_mosaic.shape[0]
            btc=0
            for btc in (range(math.ceil(num_images / batch_size))):
                # print(btc)
                if (btc + 1) * batch_size > num_images:
                    # print("last batch")
                    x = list_images_layer[btc * batch_size: num_images]

                else:
                    x = list_images_layer[btc * batch_size: (btc + 1) * batch_size]
                images_batch = process_col_images_in_batches(image_folder, x,top_colors[0],top_colors[1])
                col_images.extend(images_batch)
                if btc == 0:
                    col_neuron_mosaic = create_patch_mosaic_for_activations(images_batch, layer.receptive_field_Kernel,
                                                                       layer.receptive_field_Padding,
                                                                       layer.receptive_field_Stride, Nefesi_model.model,
                                                                       preproces_VGG16_3, layernum,[i])
                    print(layer_mosaic[:, 0])
                else:
                    next_layer_mosaic = create_patch_mosaic_for_activations(images_batch, layer.receptive_field_Kernel,
                                                                            layer.receptive_field_Padding,
                                                                            layer.receptive_field_Stride, Nefesi_model.model,
                                                                            preproces_VGG16_3, layernum,[i])
                    col_neuron_mosaic = np.concatenate((col_neuron_mosaic, next_layer_mosaic), axis=0)
                    del images_batch, x
                    torch.cuda.empty_cache()

            col_activations = top_k_elements(col_neuron_mosaic, 5)
            col_act_values = {k: float(v) for k, v in col_activations.items()}
            col_keysList = list(col_activations.keys())
            #print(list_images_layer[col_keysList[0]])
            #print(col_activations)
            #print(col_act_values)
            print("colored_activations",col_activations)
            rows = 5
            cols = 7
            aux_params = np.empty((rows, cols), dtype=int)
            for j, ind in enumerate(col_keysList):
                if list_images_layer[ind][15] == '.':
                    ind_1, ind_2, ind_3, ind_4, ind_5 = get_parameters(list_images_layer[ind])
                    aux_params[j] = [int(ind_1), int(ind_2), int(ind_3), int(ind_4), int(ind_5), 33,33]
                else:
                    ind_1, ind_2, ind_3, ind_4, ind_5, ind_6,ind_7 = get_parameters(list_images_layer[ind])
                    aux_params[j] = [int(ind_1), int(ind_2), int(ind_3), int(ind_4), int(ind_5), int(ind_6),int(ind_7)]
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            axs[0, 0].imshow(patch)
            axs[0, 0].set_title(f"Original Activation: {orig_act:.3f}")
            axs[0, 1].imshow(col_images[col_keysList[0]])
            if aux_params[0][5] == 33: # sigmoid
                params_1 = parameters_matrix[
                    aux_params[0][0], aux_params[0][1], aux_params[0][2], aux_params[0][3], aux_params[0][4]]
            else:
                params_1 = gaussian_parameters_matrix[
                    aux_params[0][0], aux_params[0][1], aux_params[0][2], aux_params[0][3], aux_params[0][4],
                    aux_params[0][5], aux_params[0][6]]
            axs[0, 1].set_title(f"Activation 1: {col_act_values[col_keysList[0]]:.3f}")
            axs[0, 1].text(0.5, -0.15, f"{params_1}", transform=axs[0, 1].transAxes, ha='center',
                           fontsize=6, fontstyle='italic')
            if aux_params[1][5] == 33:
                params_2 = parameters_matrix[
                    aux_params[1][0], aux_params[1][1], aux_params[1][2], aux_params[1][3], aux_params[1][4]]
            else:
                params_2 = gaussian_parameters_matrix[
                    aux_params[1][0], aux_params[1][1], aux_params[1][2], aux_params[1][3], aux_params[1][4],
                    aux_params[1][5], aux_params[1][6]]

            axs[0, 2].imshow(col_images[col_keysList[1]])
            axs[0, 2].set_title(f"Activation 2: {col_act_values[col_keysList[1]]:.3f}")
            axs[0, 2].text(0.5, -0.15, f"{params_2}", transform=axs[0, 2].transAxes, ha='center',
                           fontsize=6, fontstyle='italic')
            axs[1, 0].imshow(col_images[col_keysList[2]])
            if aux_params[2][5] == 33:
                params_3 = parameters_matrix[
                    aux_params[2][0], aux_params[2][1], aux_params[2][2], aux_params[2][3], aux_params[2][4]]
            else:
                params_3 = gaussian_parameters_matrix[
                    aux_params[2][0], aux_params[2][1], aux_params[2][2], aux_params[2][3], aux_params[2][4],
                    aux_params[2][5],aux_params[2][6]]
            axs[1, 0].set_title(f"Activation 3: {col_act_values[col_keysList[2]]:.3f}")
            axs[1, 0].text(0.5, -0.15, f"{params_3}", transform=axs[1, 0].transAxes, ha='center',
                           fontsize=6, fontstyle='italic')
            axs[1, 1].imshow(col_images[col_keysList[3]])
            if aux_params[3][5] == 33:
                params_4 = parameters_matrix[
                    aux_params[3][0], aux_params[3][1], aux_params[3][2], aux_params[3][3], aux_params[3][4]]
            else:
                params_4 = gaussian_parameters_matrix[
                    aux_params[3][0], aux_params[3][1], aux_params[3][2], aux_params[3][3], aux_params[3][4],
                    aux_params[3][5],aux_params[3][6]]
            axs[1, 1].set_title(f"Activation 4: {col_act_values[col_keysList[3]]:.3f}")
            axs[1, 1].text(0.5, -0.15, f"{params_4}", transform=axs[1, 1].transAxes, ha='center',
                           fontsize=6, fontstyle='italic')
            axs[1, 2].imshow(col_images[col_keysList[4]])
            if aux_params[4][5] == 33:
                params_5 = parameters_matrix[
                    aux_params[4][0], aux_params[4][1], aux_params[4][2], aux_params[4][3], aux_params[4][4]]
            else:
                params_5 = gaussian_parameters_matrix[
                    aux_params[4][0], aux_params[4][1], aux_params[4][2], aux_params[4][3], aux_params[4][4],
                    aux_params[4][5],aux_params[4][6]]
            axs[1, 2].set_title(f"Activation 5: {col_act_values[col_keysList[4]]:.3f}")
            axs[1, 2].text(0.5, -0.15, f"{params_5}", transform=axs[1, 2].transAxes, ha='center',
                           fontsize=6, fontstyle='italic')
            fig.suptitle(f"Layer:{layernum} Neuron:{i} Color Selectivity: {round(col_sel, 2)}")
            grid_acc = col_act_values[col_keysList[0]] / orig_act
            grid_activation = col_act_values[col_keysList[0]]


        else:

            rows = 5
            cols = 7
            aux_params = np.empty((rows, cols), dtype=int)
            for j, ind in enumerate(keysList):
                print(ind)
                if list_images_layer[ind][15] == '.':
                    ind_1, ind_2, ind_3, ind_4, ind_5 = get_parameters(list_images_layer[ind])
                    aux_params[j] = [int(ind_1), int(ind_2), int(ind_3), int(ind_4), int(ind_5),33,33]

                else:
                    ind_1, ind_2, ind_3, ind_4, ind_5, ind_6,ind_7 = get_parameters(list_images_layer[ind])
                    aux_params[j] = [int(ind_1), int(ind_2), int(ind_3), int(ind_4), int(ind_5), int(ind_6),int(ind_7)]
                print(aux_params[j])
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            axs[0, 0].imshow(patch)
            axs[0, 0].set_title(f"Original Activation: {orig_act:.3f}")
            axs[0, 1].imshow(Image.open(os.path.join(image_folder, list_images_layer[int(keysList[0])])).convert("RGB"))
            if aux_params[0][5] == 33:
                params_1 = parameters_matrix[aux_params[0][0], aux_params[0][1], aux_params[0][2], aux_params[0][3], aux_params[0][4]]
            else:
                params_1 = gaussian_parameters_matrix[
                    aux_params[0][0], aux_params[0][1], aux_params[0][2], aux_params[0][3], aux_params[0][4], aux_params[0][5],aux_params[0][6]]

            axs[0, 1].set_title(f"Activation 1: {activations[int(keysList[0])]:.3f}")
            axs[0, 1].text(0.5, -0.15, f"{params_1}", transform=axs[0, 1].transAxes, ha='center',
                           fontsize=6, fontstyle='italic')
            if aux_params[1][5] == 33:
                params_2 = parameters_matrix[
                    aux_params[1][0], aux_params[1][1], aux_params[1][2], aux_params[1][3], aux_params[1][4]]
            else:
                params_2 = gaussian_parameters_matrix[
                    aux_params[1][0], aux_params[1][1], aux_params[1][2], aux_params[1][3], aux_params[1][4], aux_params[1][5],aux_params[1][6]]

            axs[0, 2].imshow(Image.open(os.path.join(image_folder, list_images_layer[int(keysList[1])])).convert("RGB"))
            axs[0, 2].set_title(f"Activation 2: {activations[int(keysList[1])]:.3f}")
            axs[0, 2].text(0.5, -0.15, f"{params_2}", transform=axs[0, 2].transAxes, ha='center',
                           fontsize=6, fontstyle='italic')
            axs[1, 0].imshow(Image.open(os.path.join(image_folder, list_images_layer[int(keysList[2])])).convert("RGB"))
            if aux_params[2][5] == 33:
                params_3 = parameters_matrix[
                    aux_params[2][0], aux_params[2][1], aux_params[2][2], aux_params[2][3], aux_params[2][4]]
            else:
                params_3 = gaussian_parameters_matrix[
                    aux_params[2][0], aux_params[2][1], aux_params[2][2], aux_params[2][3], aux_params[2][4], aux_params[2][5],aux_params[2][6]]
            axs[1, 0].set_title(f"Activation 3: {activations[int(keysList[2])]:.3f}")
            axs[1, 0].text(0.5, -0.15, f"{params_3}", transform=axs[1, 0].transAxes, ha='center',
                           fontsize=6, fontstyle='italic')
            axs[1, 1].imshow(Image.open(os.path.join(image_folder, list_images_layer[int(keysList[3])])).convert("RGB"))
            if aux_params[3][5] == 33:
                params_4 = parameters_matrix[
                    aux_params[3][0], aux_params[3][1], aux_params[3][2], aux_params[3][3], aux_params[3][4]]
            else:
                params_4 = gaussian_parameters_matrix[
                    aux_params[3][0], aux_params[3][1], aux_params[3][2], aux_params[3][3], aux_params[3][4], aux_params[3][5],aux_params[3][6]]
            axs[1, 1].set_title(f"Activation 4: {activations[int(keysList[3])]:.3f}")
            axs[1, 1].text(0.5, -0.15, f"{params_4}", transform=axs[1, 1].transAxes, ha='center',
                           fontsize=6, fontstyle='italic')
            axs[1, 2].imshow(Image.open(os.path.join(image_folder, list_images_layer[int(keysList[4])])).convert("RGB"))
            if aux_params[4][5]== 33:
                params_5 = parameters_matrix[
                    aux_params[4][0], aux_params[4][1], aux_params[4][2], aux_params[4][3], aux_params[4][4]]
            else:
                params_5 = gaussian_parameters_matrix[
                    aux_params[4][0], aux_params[4][1], aux_params[4][2], aux_params[4][3], aux_params[4][4], aux_params[4][5],aux_params[4][6]]
            axs[1, 2].set_title(f"Activation 5: {activations[int(keysList[4])]:.3f}")
            axs[1, 2].text(0.5, -0.15, f"{params_5}", transform=axs[1, 2].transAxes, ha='center',
                           fontsize=6, fontstyle='italic')
            fig.suptitle(f"Layer:{layernum} Neuron:{i} Color Selectivity: {round(col_sel,2)}")
            grid_acc = activations[keysList[0]] / orig_act
            grid_activation = activations[keysList[0]]


        plt.tight_layout()
        grid_accuracy[layer_idx][i] = grid_acc
        if show_images:
            plt.show()
        if save_images:
            save_dir = f"/home/vbenito/PycharmProjects/pythonProject/final_activations/layer_{layernum}/neuron_{i}"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"Layer_{layernum}_Neuron_{i}_activation.png")) #first plot
        plt.close(fig)

        initial_temperature = annealing_params[0]
        cooling_rate = annealing_params[1]
        iterations = annealing_params[2]
        n_values = [3, 5, 10, 14, 24, 32, 40, 60, 76, 92, 132, 164, 196]
        n = n_values[layer_idx]
        params_list = [list(params_1.values()), list(params_2.values()), list(params_3.values()), list(params_4.values()), list(params_5.values())]
        #print(params_list)
        best_results = []
        original_images = []
        # Iterate over each set of parameters
        idx_p = 0
        for params in params_list:
            if len(params) == 5:
                if col_sel > 0.2:
                     original_params = np.array([params[0], params[1], params[2], params[3], params[4], top_colors[0], top_colors[1]],dtype=object)
                     #img = calculate_image(n=n, a=original_params[0], b=original_params[1], theta=original_params[2],dx=original_params[3], dy=original_params[4]).convert("RGB")
                     #col_img = map_color(img, original_params[5], original_params[6])
                     col_img = col_images[col_keysList[idx_p]]

                else:
                     original_params = np.array([params[0], params[1], params[2], params[3], params[4], [0, 0, 0],[255, 255, 255]],dtype=object)
                     #img = calculate_image(n=n, a=original_params[0], b=original_params[1], theta=original_params[2],dx=original_params[3], dy=original_params[4]).convert("RGB")
                     #col_img = img
                     col_img = Image.open(os.path.join(image_folder, list_images_layer[int(keysList[idx_p])])).convert("RGB")

            else:
                if col_sel > 0.2:
                    original_params = np.array([params[0], params[1], params[2], params[3], params[4], params[5],params[6], top_colors[0],top_colors[1]], dtype=object)
                    #img = calculate_gaussian_image(n=n, mu_x=original_params[0], mu_y=original_params[1],sigma_x=original_params[2], sigma_y=original_params[3],theta=original_params[4], dx=original_params[5],dy=original_params[6]).convert("RGB")
                    #col_img = map_color(img, original_params[7], original_params[8])
                    col_img = col_images[col_keysList[idx_p]]
                else:
                    original_params = np.array([params[0], params[1], params[2], params[3], params[4], params[5],params[6], [0, 0, 0],[255, 255, 255]], dtype=object)
                    #img = calculate_gaussian_image(n=n, mu_x=original_params[0], mu_y=original_params[1],sigma_x=original_params[2], sigma_y=original_params[3],theta=original_params[4], dx=original_params[5],dy=original_params[6]).convert("RGB")
                    #col_img = img
                    col_img = Image.open(os.path.join(image_folder, list_images_layer[int(keysList[idx_p])])).convert("RGB")

            best_params, history, best_img = simulated_annealing_with_derivatives(initial_temperature=initial_temperature,cooling_rate=cooling_rate, iterations=iterations, best_params=original_params,layer=layer,layernum=layernum,neuron=i, model=Nefesi_model.model,n=n,col_sel=col_sel)
            best_activation = history[-1][0]
            best_results.append((best_params, best_activation, best_img))
            original_images.append((original_params, col_img))
            idx_p += 1

        # Find the best overall parameters
        best_overall_params, best_overall_activation, best_overall_img = max(best_results, key=lambda x: x[1])

        # Plot the original and best overall images side by side
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))

        # Top Scoring
        axes[0].imshow(patch)
        axes[0].set_title(f"Top Scoring Activation {orig_act:.3f}", fontsize=12)

        # Original image (using the first set of params)
        original_params, col_img = original_images[0]
        axes[1].imshow(col_img)
        axes[1].set_title(f"Grid Activation {grid_activation:.3f}", fontsize=12)
        axes[1].set_xlabel(f"{format_params(original_params[:-2])}", fontsize=6)

        # Best overall image after annealing
        axes[2].imshow(best_overall_img)
        axes[2].set_title(f"Annealing Activation: {best_overall_activation[0]:.2f}", fontsize=12)
        axes[2].set_xlabel(f"{format_params(best_overall_params[:-2])}", fontsize=6)
        fig.suptitle(f"Layer_{layernum}_Neuron_{i} Activations Resume")
        # Adjust layout and show plot
        plt.tight_layout()

        if show_images:
            plt.show()
        if save_images:
            plt.savefig(os.path.join(save_dir, f"Layer_{layernum}_Neuron_{i}_final_activation.png")) #second plot
        plt.close(fig)
        ann_acc = best_overall_activation / orig_act
        annealing_accuracy[layer_idx][i] = ann_acc
        best_images_matrix[layer_idx][i] = (best_overall_img, best_overall_params)
        if col_sel >0.2:
            del col_images
        torch.cuda.empty_cache()
    np.save(os.path.join("/data/134-1/datasets/final_dataset", f"best_images_matrix.npy"), best_images_matrix)
    np.save(os.path.join("/data/134-1/datasets/final_dataset", f"annealing_accuracy.npy"), annealing_accuracy)
    np.save(os.path.join("/data/134-1/datasets/final_dataset", f"grid_accuracy.npy"), grid_accuracy)

def preproces_VGG162(imgs_hr):
    # img=np.array(imgs_hr)

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),

    ])

    tnsr = [preprocess(imgs_hr)]

    return tnsr


def create_patch_mosaic_for_activations(list_images, K, P, S, model, preprocessing, layer_name, neuron_list=None):
    """Returns a list of activations produced by the input images in list_images on the neurons in neuron_list.
        :param list_images: List of PIL images of size K.
        :param K: Int with the size of the kernel of the layer.
        :param P: Int with the size of the padding of the layer.
        :param S: Int with the size of the stride of the layer.
        :param model: deepmodel of nefesi
        :param preprocessing: preprocessing function of the model, WITHOUT RESHAPING!!.
        :param layer_name: String with the layer name in the model
        :param neuron_list: list of Ints with the neurons that we want to get the activations from. If empty, the function will return activations in all neurons

       :return: Numpy array of [I,N] size that contains the images the activation in neurons N in neuron list for all the images I in list_images.
       """

    total_images = len(list_images)

    if int(np.sqrt(total_images)) == np.sqrt(total_images):
        squarex = int(np.sqrt(total_images))
        squarey = int(np.sqrt(total_images))

    else:

        squarex = int(np.sqrt(total_images)) + 1
        squarey = int(np.sqrt(total_images))

    margin = S - P % S
    if margin >= S:
        margin = 0

    distance_next = S - K % S
    if distance_next >= S:
        distance_next = 0

    X_size = int(margin + K * squarex + distance_next * (squarex - 1))
    Y_size = int(margin + K * squarey + distance_next * (squarey - 1))

    if X_size < 224:
        X_size = 224
    if Y_size < 224:
        Y_size = 224

    super_image = Image.new('RGB', (X_size, Y_size))

    for y in range(squarey):
        for x in range(squarex):
            if x + y * squarex < total_images:
                imatge = list_images[x + y * squarex]
                paste_x = margin + x * (K + distance_next)
                paste_y = margin + y * (K + distance_next)
                super_image.paste(imatge, (paste_x, paste_y))
    # plt.imshow(super_image)
    # plt.show()

    super_image = preprocessing(super_image)
    super_image = [torch.unsqueeze(i, 0) for i in super_image]
    activations = model.calculate_activations(layer_name, super_image)

    if neuron_list == None:
        final_activations = np.zeros((total_images, activations[0].shape[-1]))

        for y in range(squarey):
            for x in range(squarex):
                if x + y * squarex < total_images:
                    position_x = math.ceil(P / S) + math.ceil((K + distance_next) / S * x)
                    position_y = math.ceil(P / S) + math.ceil((K + distance_next) / S * y)

                    final_activations[int(x + y * squarex), :] = activations[0][0, position_y, position_x, :]

    else:
        final_activations = np.zeros((total_images, len(neuron_list)))

        for y in range(squarey):
            for x in range(squarex):
                if x + y * squarex < total_images:
                    position_x = math.ceil(P / S) + math.ceil((K + distance_next) / S * x)
                    position_y = math.ceil(P / S) + math.ceil((K + distance_next) / S * y)

                    final_activations[int(x + y * squarex), :] = activations[0][0, position_y, position_x, neuron_list]

    return final_activations

def map_color(image, col1, col2):
    # Convert image to grayscale
    grayscale_image = image.convert('L')

    # Convert colors to HSV format
    hsv_col1 = colorsys.rgb_to_hsv(*[x / 255.0 for x in col1])
    hsv_col2 = colorsys.rgb_to_hsv(*[x / 255.0 for x in col2])

    # Get the grayscale values
    grayscale_values = np.array(grayscale_image)

    # Normalize grayscale values to [0, 1]
    normalized_values = grayscale_values / 255.0

    # Interpolate between the two colors in HSV space
    interpolated_hsv = (1 - normalized_values[:, :, None]) * np.array(hsv_col1) + normalized_values[:, :,
                                                                                  None] * np.array(hsv_col2)

    # Convert interpolated HSV values back to RGB
    interpolated_rgb = np.array([[colorsys.hsv_to_rgb(*pixel) for pixel in row] for row in interpolated_hsv])

    # Convert RGB values to uint8
    interpolated_rgb = np.clip(interpolated_rgb * 255, 0, 255).astype(np.uint8)

    # Create the final colorized image
    colorized_image = Image.fromarray(interpolated_rgb, 'RGB')

    return colorized_image


def scan_topColors(img, n=2):
    # Convert the image to RGB
    img_rgb = img.convert('RGB')

    # Get the pixel values of the image
    pixel_values = np.array(img_rgb)

    # Flatten the pixel values array
    pixel_values_flat = pixel_values.reshape(-1, pixel_values.shape[-1])

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n,n_init=10)
    kmeans.fit(pixel_values_flat)

    # Get the cluster centers (i.e., the top colors)
    top_colors = kmeans.cluster_centers_.astype(int)

    return top_colors

def preproces_Resnet(imgs_hr):

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tnsr = [preprocess(imgs_hr)]

    return tnsr


def preproces_Resnet2(imgs_hr):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    tnsr = [preprocess(imgs_hr)]

    return tnsr


def perturb_params(best_params, col_sel):
    neighbor_params = best_params.copy()
    if len(best_params) == 7:
        perturb_range = np.array([0.1, 0.1, 1, 1, 1])  # Example range for perturbation
        neighbor_params[:-2] += np.random.uniform(-perturb_range, perturb_range)
        neighbor_params[0] = abs(neighbor_params[0])
        neighbor_params[2] = int(neighbor_params[2]) % 360
        neighbor_params[3] = int(neighbor_params[3])
        neighbor_params[4] = int(neighbor_params[4])
    else:
        perturb_range = np.array([0.1, 0.1, 0.1, 0.1, 1, 1, 1])  # Example range for perturbation
        neighbor_params[:-2] += np.random.uniform(-perturb_range, perturb_range)
        neighbor_params[2] = int(neighbor_params[2]) % 180
        neighbor_params[5] = int(neighbor_params[5])
        neighbor_params[6] = int(neighbor_params[6])
        if neighbor_params[2] < neighbor_params[3]:
            neighbor_params[2] = neighbor_params[3] + 0.01

    if col_sel > 0.2:
        neighbor_params[-2] = tuple(np.clip(np.array(neighbor_params[-2]) + np.random.randint(-5, 5, size=3), 0, 255))
        neighbor_params[-1] = tuple(np.clip(np.array(neighbor_params[-1]) + np.random.randint(-5, 5, size=3), 0, 255))


    return neighbor_params


def simulated_annealing_with_derivatives(initial_temperature, cooling_rate, iterations, best_params, layer, layernum,neuron,model,n,col_sel):
    current_params = np.array(best_params,dtype=object)
    current_temperature = initial_temperature
    print(f"Layer:{layernum}, Neuron: {neuron}, init parameters: {best_params}")
    best_solutions_history = []

    for iter in range(iterations):
        # Generate a neighboring solution
        # Perturb parameters within a certain range
        neighbor_params = perturb_params(best_params, col_sel)

        # Evaluate the objective function for current and neighbor solutions
        current_objective, curr_img = objective_function(current_params, layer, layernum,  neuron, model,n,col_sel)
        #if iter ==0:
            #print(current_params)
            #print(current_objective)
        neighbor_objective,nei_img = objective_function(neighbor_params, layer, layernum,  neuron, model,n,col_sel)

        # Decide whether to accept the neighbor as the new current solution
        acceptance_condition = neighbor_objective > current_objective or np.random.rand() < np.exp((neighbor_objective - current_objective) / current_temperature)

        if acceptance_condition:
            current_params = neighbor_params
            current_objective = neighbor_objective  # Update current_objective with the neighbor's objective
            curr_img = nei_img

        # Update the best solution
        best_solution, best_img = objective_function(best_params, layer, layernum, neuron, model,n,col_sel)
        if current_objective > best_solution:
            best_params = np.array(current_params)
            best_img = curr_img


        best_solutions_history.append(best_solution)

        # Cool down the temperature
        current_temperature *= cooling_rate
    """"
    if len(best_params) ==7:
        print(f"    Best solution: {best_solution}, a: {best_params[0]}, b: {best_params[1]}, theta: {best_params[2]}, dx: {best_params[3]}, dy: {best_params[4]}, col1: {best_params[5]}, col2: {best_params[6]}\n")
    else:
        print(f"    Best solution: {best_solution}, mu_x: {best_params[0]}, mu_y: {best_params[1]}, sigma_x: {best_params[2]}, sigma_y: {best_params[3]}, theta: {best_params[4]}, dx: {best_params[5]}, dy: {best_params[6]}, col1:{best_params[7]}, col2:{best_params[8]}\n")
    """
    return best_params, best_solutions_history, best_img

def objective_function(parameters, layer, layernum,  neuron, model,n,sel_col):
    if len(parameters) ==7:
        a , b, theta, dx, dy, col1, col2= parameters
        image = calculate_image(n=n, a=a, b=b, theta=theta, dx=dx, dy=dy, ax=None, show=False).convert('RGB')
        if sel_col > 0.2:
            col_img = map_color(image, col1, col2)
        else:
            col_img = image
        max_act = create_patch_mosaic_for_activations([col_img], layer.receptive_field_Kernel, layer.receptive_field_Padding,layer.receptive_field_Stride, model,preproces_VGG16_3, layernum, neuron_list=[neuron])

        return max_act, col_img
    else:
        mu_x, mu_y, sigma_x, sigma_y, theta, dx, dy, col1, col2 = parameters
        image = calculate_gaussian_image(n=n, mu_x=mu_x, mu_y=mu_y, sigma_x=sigma_x,sigma_y=sigma_y,theta=theta, dx=dx,dy=dy, ax=None, show=False).convert('RGB')
        if sel_col > 0.2:
            col_img = map_color(image, col1, col2)
        else:
            col_img = image
        max_act= create_patch_mosaic_for_activations([col_img], layer.receptive_field_Kernel, layer.receptive_field_Padding,layer.receptive_field_Stride, model, preproces_VGG16_3, layernum, neuron_list=[neuron])
        return max_act, col_img


def format_params(params):
    sigmoid_names = ['a', 'b', 'theta', 'dx', 'dy']
    gaussian_names = ['mu_x','mu_y','sigma_x','sigma_y','theta','dx','dy']
    if len(params)==5:
        formatted_params = [f"{name}={round(params[i], 2)}" for i, name in enumerate(sigmoid_names)]
    else:
        formatted_params = [f"{name}={round(params[i], 2)}" for i, name in enumerate(gaussian_names)]
    return ', '.join(formatted_params)

def sigmoid(x, y, a, b, theta=0):
    theta -= 45
    rotated_x = x * np.cos(np.radians(theta)) - (y - b) * np.sin(np.radians(theta))
    rotated_y = x * np.sin(np.radians(theta)) + (y - b) * np.cos(np.radians(theta))
    return 1 / (1 + np.exp(-a * (rotated_x + rotated_y)))


def derivative(image, dx=0, dy=0, dtype='sobel'):
    input_dtype = image.dtype
    dx = int(dx)
    dy = int(dy)

    pad_size = dx + dy
    n = image.shape[0]
    new_size = ((n + 2 * pad_size, n + 2 * pad_size))

    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    padded_image = resized_image

    if dtype == 'scharr':
        for _ in range(dx):
            padded_image = cv2.Scharr(padded_image, cv2.CV_64F, 1, 0)
        for _ in range(dy):
            padded_image = cv2.Scharr(padded_image, cv2.CV_64F, 0, 1)
    elif dtype == 'sobel':
        for _ in range(dx):
            padded_image = cv2.Sobel(padded_image, cv2.CV_64F, 1, 0)
        for _ in range(dy):
            padded_image = cv2.Sobel(padded_image, cv2.CV_64F, 0, 1)

    cropped_image = padded_image[pad_size:-pad_size, pad_size:-pad_size]
    cropped_image = cv2.normalize(cropped_image, None, 0, 1, cv2.NORM_MINMAX)

    return cropped_image


def sigmoid_matrix(n=3, a=1, b=0, theta=0, dx=0, dy=0, dtype='sobel'):
    x_range = np.linspace(-5, 5, int(n))
    y_range = np.linspace(-5, 5, int(n))

    x, y = np.meshgrid(x_range, y_range)

    z = sigmoid(x, y, a, b, theta)

    if dx or dy:
        z = derivative(z, dx, dy, dtype)

    return z


def calculate_image(n=3, a=1, b=0, theta=0, dx=0, dy=0, dtype='sobel', ax=None, show=False):
    sigmoid_values = sigmoid_matrix(n, a, b, theta, dx, dy, dtype)

    # Normalize values to range [0, 1]
    min_val = np.min(sigmoid_values)
    max_val = np.max(sigmoid_values)
    denominator = max_val - min_val if max_val != min_val else 1e-6  # Add a small epsilon to avoid division by zero
    sigmoid_values = (sigmoid_values - min_val) / denominator
    # Scale values to 0-255 for PIL Image
    scaled_values = (sigmoid_values * 255).astype(np.uint8)

    # Create a PIL Image from the matrix
    image = Image.fromarray(scaled_values, mode='L')  # mode='L' for grayscale

    if show:
        if ax is None:
            fig, ax = plt.subplots()
            ax.axis('off')  # Turn off axes if subplot is created
        # Display the image without axes
        ax.imshow(image, cmap='gray')
        ax.axis('off')  # Turn off axes

    return image

from PIL import Image


def gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y, theta=0):
    if sigma_x == sigma_y:
        exponent = -((x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2))
    else:
        rotated_x = (x - mu_x) * np.cos(np.radians(theta)) - (y - mu_y) * np.sin(np.radians(theta))
        rotated_y = (x - mu_x) * np.sin(np.radians(theta)) + (y - mu_y) * np.cos(np.radians(theta))
        exponent = -((rotated_x - mu_x) ** 2 / (2 * sigma_x ** 2) + (rotated_y - mu_y) ** 2 / (2 * sigma_y ** 2))

    return np.exp(exponent)


"""
x: The x-coordinate of the point at which the Gaussian function is evaluated.
y: The y-coordinate of the point at which the Gaussian function is evaluated.
mu_x: The mean or center of the Gaussian along the x-axis.
mu_y: The mean or center of the Gaussian along the y-axis.
sigma_x: The standard deviation of the Gaussian along the x-axis. It controls the spread or width of the Gaussian along the x-axis.
sigma_y: The standard deviation of the Gaussian along the y-axis. It controls the spread or width of the Gaussian along the y-axis.
amplitude: The amplitude or height of the Gaussian. It controls the overall intensity or magnitude of the Gaussian function.
"""


def gaussian_matrix(n=3, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1, theta=0, dx=0, dy=0, dtype='sobel'):
    x_range = np.linspace(-5, 5, int(n))
    y_range = np.linspace(-5, 5, int(n))

    x, y = np.meshgrid(x_range, y_range)

    z = gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y, theta)

    if dx or dy:
        z = derivative(z, dx, dy, dtype)

    return z


def calculate_gaussian_image(n=3, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1, theta=0, dx=0, dy=0, dtype='sobel', ax=None,
                             show=False):
    gaussian_values = gaussian_matrix(n, mu_x, mu_y, sigma_x, sigma_y, theta, dx, dy, dtype)

    # Normalize values to range [0, 1]
    min_val = np.min(gaussian_values)
    max_val = np.max(gaussian_values)
    denominator = max_val - min_val if max_val != min_val else 1e-6  # Add a small epsilon to avoid division by zero
    normalized_values = (gaussian_values - min_val) / denominator

    # Scale values to 0-255 for PIL Image
    scaled_values = (normalized_values * 255).astype(np.uint8)

    # Create a PIL Image from the matrix
    image = Image.fromarray(scaled_values, mode='L')  # mode='L' for grayscale

    if show:
        if ax is None:
            fig, ax = plt.subplots()
            ax.axis('off')  # Turn off axes if subplot is created
        # Display the image without axes
        ax.imshow(image, cmap='gray')
        ax.axis('off')  # Turn off axes

    return image


def main():
    """
    path_images = '/data/134-1/datasets/ImageNetFused/'
    Nefesi_model = NetworkData.load_from_disk('/home/guillem/Nefesi2022/nefesi/PaperClip/Nefesi_models/CLIP/ColorCLIP.obj')
    dataset = ImageDataset(src_dataset=path_images,target_size=(224,224),preprocessing_function=preproces_Resnet,color_mode='rgb')
    dataset.src_segmentation_dataset = '/data/135-1/datasets/ImageNetFusedSegmented/'
    dataset = ImageDataset(src_dataset=path_images, target_size=(224, 224), preprocessing_function=preproces_Resnet2,
                           color_mode='rgb')
    device = 0
    model, preprocess = clip.load('RN50', device)
    model = model.visual

    Nefesi_model.dataset = dataset
    file_name = 'Normal_Retrained'

    weights_location = '/home/guillem/Nefesi2022/nefesi/Retrain_Force_Selectivity/' + file_name

    model = vgg16()
    model.to(device)
    weights = torch.load(weights_location, map_location=torch.device(device))
    model.load_state_dict(weights)

    deepmodel = DeepF.deep_model(model.features)

    Nefesi_model.model = deepmodel
    """
    Nefesi_model = NetworkData.load_from_disk(
        '/home/guillem/Nefesi2022/nefesi/Nefesi_models/VGG16/1-Normal_Retrained.obj')
    Path_images = '/data/134-1/datasets/ImageNetFused/'
    dataset = ImageDataset(src_dataset=Path_images, target_size=(224, 224), preprocessing_function=preproces_VGG162,
                           color_mode='rgb')
    Nefesi_model.dataset = dataset
    file_name = 'Normal_Retrained'
    device = 0
    weights_location = '/home/guillem/Nefesi2022/nefesi/Retrain_Force_Selectivity/' + file_name

    model = vgg16()
    model.to(device)
    weights = torch.load(weights_location, map_location=torch.device(device))
    model.load_state_dict(weights)

    deepmodel = DeepF.deep_model(model.features)

    Nefesi_model.model = deepmodel


    # Path to the images used to calculate neuron activations


    # Get neuron
    layernum = '12'
    neuron = Nefesi_model.get_neuron_of_layer('features.' + layernum, 0)
    print(neuron)
    layer = Nefesi_model.get_layer_by_name('features.' + layernum)

    # We select a top scoring patch as an example of input image
    number_of_top_scoring_image = 0

    #patch = Image.fromarray((dataset.get_patch(img_name=neuron.images_id[number_of_top_scoring_image],crop_pos= neuron.xy_locations[number_of_top_scoring_image], K=layer.receptive_field_Kernel,P= layer.receptive_field_Padding, S=layer.receptive_field_Stride, mode='constant') *255).astype(np.uint8))
    layer_idx = 5
    image_folder = f"/data/134-1/datasets/final_dataset/{layer_idx}"

    list_images_layer = [file_name for file_name in os.listdir(image_folder) if file_name.endswith(".jpg")]
    list_images_layer = tuple(list_images_layer)
    #Image.open(os.path.join(image_folder, file_name)).convert("RGB")
    parameters_matrix = np.load(f'/data/134-1/datasets/final_dataset/parameters_matrix_{layer_idx}.npy', allow_pickle=True)
    gaussian_parameters_matrix = np.load(f'/data/134-1/datasets/final_dataset/gaussian_parameters_matrix_{layer_idx}.npy',allow_pickle=True)
    #non_none_count = np.count_nonzero(parameters_matrix != None)
    #print("Number of non-None elements:", non_none_count)
    batch_size = 10
    # Concatenate the batches along with their parameter indices
    num_images = len(list_images_layer)
    print(num_images)
    #images = []
    for btc in (range(math.ceil(num_images/batch_size))):
        #print(btc)
        if (btc+1)*batch_size > num_images:
            #print("last batch")
            x = list_images_layer[btc * batch_size: num_images]

        else:
            x = list_images_layer[btc*batch_size: (btc+1)*batch_size]
        images_batch = process_images_in_batches(image_folder, x)
        #images.extend(images_batch)

        if btc == 0:
            layer_mosaic = create_patch_mosaic_for_activations(images_batch, layer.receptive_field_Kernel,layer.receptive_field_Padding,layer.receptive_field_Stride, deepmodel,preproces_VGG16_3, layernum)
            print(layer_mosaic[:,0])
        else:
            next_layer_mosaic = create_patch_mosaic_for_activations(images_batch, layer.receptive_field_Kernel,layer.receptive_field_Padding,layer.receptive_field_Stride, deepmodel,preproces_VGG16_3, layernum)
            layer_mosaic = np.concatenate((layer_mosaic, next_layer_mosaic), axis=0)
            del images_batch, x
            torch.cuda.empty_cache()





    #print(list_images_layer[18159])
    #plt.imshow(images[18159])
    #plt.show()
    #layer_mosaic = create_patch_mosaic_for_activations(images, layer.receptive_field_Kernel,
                                                       #layer.receptive_field_Padding, layer.receptive_field_Stride,
                                                       #deepmodel, preproces_VGG16_3, layernum)
    # Cleanup after processing
    #print(layer_mosaic[18159, 1])
    #thing=create_patch_mosaic_for_activations([patch,patch,patch,patch,patch,patch,patch,patch,patch,patch],layer.receptive_field_Kernel,layer.receptive_field_Padding,layer.receptive_field_Stride,deepmodel,preproces_VGG16_3,layernum)
    #layer_mosaic = create_patch_mosaic_for_activations(list_images_layer,layer.receptive_field_Kernel,layer.receptive_field_Padding,layer.receptive_field_Stride,deepmodel,preproces_VGG16_3,layernum)

    top_scoring_image6_id = neuron.images_id[number_of_top_scoring_image] # Extract the top scoring image 6
    print(layer_mosaic.shape)
    #save the best images in a matrix
    new = False
    if new:
        num_layers = 13
        max_neurons_per_layer = 512
        best_images_matrix = np.empty((num_layers, max_neurons_per_layer), dtype=object)
        grid_accuracy = np.empty((num_layers,max_neurons_per_layer),dtype=float)
        annealing_accuracy = np.empty((num_layers,max_neurons_per_layer),dtype=float)
    # Assuming you have appropriate data for list_images_layer_0, Nefesi_model, and layernum
    else:
        best_images_matrix = np.load(f'/data/134-1/datasets/final_dataset/best_images_matrix.npy',allow_pickle=True)
        grid_accuracy= np.load(f'/data/134-1/datasets/final_dataset/grid_accuracy.npy',allow_pickle=True)
        annealing_accuracy = np.load(f'/data/134-1/datasets/final_dataset/annealing_accuracy.npy', allow_pickle=True)
    temperature = 2000
    cooling_rate = 0.95
    iterations = 500
    annealing_params = [temperature,cooling_rate,iterations]




    plot_layer_mosaic(layer_mosaic, list_images_layer, Nefesi_model, parameters_matrix ,gaussian_parameters_matrix,annealing_params,layer,layernum,layer_idx, image_folder, best_images_matrix, grid_accuracy, annealing_accuracy, show_images= False, save_images=True)

    '''
    Grey
    hue
    activation
    comparison
    '''


if __name__ == '__main__':
    main()
