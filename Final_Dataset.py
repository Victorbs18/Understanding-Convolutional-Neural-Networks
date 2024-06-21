import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from skimage.transform import rotate
from PIL import Image
import pandas as pd
import cv2


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

import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
# n_values = [3, 5, 10, 14, 24, 32, 40, 60, 76, 92, 132, 164, 196]
import os
n_values = [3, 5, 10, 14, 24, 32, 40, 60, 76, 92, 132, 164, 196]
dx_values = [0, 1, 2, 3, 4]  # Values for dx
dy_values = [0, 1, 2, 3, 4]  # Values for dy
# Directory to save images

save_dir = "/data/134-1/datasets/final_dataset"
import hashlib

# Define a function to calculate the hash of an image
def calculate_image_hash(image):
    image_bytes = np.array(image).tobytes()
    return hashlib.md5(image_bytes).hexdigest()
unique_images = {}
for i, n in enumerate(n_values):
    # Grid sizes for each layer
    #grid_size = grid_sizes[i]
    grid_size = 6

    # Define theta_values, a_values, and b_values
    theta_values_sigmoid = np.linspace(0, 360, grid_size*2, endpoint=False).astype(int)
    theta_values_gaussian = np.linspace(0, 180, grid_size, endpoint=False).astype(int)
    a_values = np.linspace(0.1, 3, int(grid_size/2))
    b_values = np.linspace(-3, 3, grid_size)
    mu_x_values = np.linspace(-5,5,grid_size)
    mu_y_values = np.linspace(-5,5,grid_size)
    sigma_x_values = np.linspace(0.1,2,grid_size)
    sigma_y_values = np.linspace(0.1,2,grid_size)

    # Create directories for each index
    folder_name = os.path.join(save_dir, str(i))
    os.makedirs(folder_name, exist_ok=True)


    parameters_matrix = np.empty((len(a_values), len(b_values), len(theta_values_sigmoid), len(dx_values), len(dy_values)), dtype=object)

    gaussian_parameters_matrix = np.empty((len(mu_x_values), len(mu_y_values), len(sigma_x_values), len(sigma_y_values),len(theta_values_gaussian), len(dx_values), len(dy_values)), dtype=object)

    # Create a set to store unique image hashes
    unique_image_hashes = set()
    hash_im_count = 0
    # Loop through each parameter combination
    for dx_idx, dx in enumerate(dx_values):
        for dy_idx, dy in enumerate(dy_values):
            for b_idx, b in enumerate(b_values):
                for theta_idx, theta in enumerate(theta_values_sigmoid):
                    for a_idx, a in enumerate(a_values):
                        image = calculate_image(n=n, a=a, b=b, theta=theta, dx=dx, dy=dy,
                                                dtype='sobel', show=False)

                        # Calculate the hash of the image
                        image_hash = calculate_image_hash(image)
                        # Check if the image hash is already in the set
                        if image_hash not in unique_image_hashes:
                            hash_im_count +=1
                            unique_image_hashes.add(image_hash)
                            unique_images[image_hash] = [{
                                'n': n,
                                'a': a,
                                'b': b,
                                'theta': theta,
                                'dx': dx,
                                'dy': dy,
                            }]

                            # Save parameters in the parameters_matrix
                            parameters_matrix[a_idx, b_idx, theta_idx, dx_idx, dy_idx] = {'a': round(a, 2), 'b': round(b, 2), 'theta': round(theta, 2), 'dx': dx,'dy': dy
                    }

                            # Save the image (modify this part as needed)
                            image_name = f"{a_idx:03d}{b_idx:03d}{theta_idx:03d}{dx_idx:03d}{dy:03d}.jpg"
                            image_path = os.path.join(folder_name, image_name)
                            image.save(image_path)
                        else:
                            unique_images[image_hash].append({
                                'n': n,
                                'a': a,
                                'b': b,
                                'theta': theta,
                                'dx': dx,
                                'dy': dy,
                            })
                            continue  # Skip if image is a duplicate

    for mu_x_idx, mu_x in enumerate(mu_x_values):
        for mu_y_idx, mu_y in enumerate(mu_y_values):
            for sigma_y_idx, sigma_y in enumerate(sigma_y_values):
                for sigma_x_idx, sigma_x in enumerate(sigma_x_values):
                    for theta_idx, theta in enumerate(theta_values_gaussian):
                        for dx_idx, dx in enumerate(dx_values):
                            for dy_idx,dy in enumerate(dy_values):
                                image = calculate_gaussian_image(n=n, mu_x=mu_x, mu_y=mu_y, sigma_x=sigma_x, sigma_y=sigma_y,theta=theta, dx=dx,dy=dy,dtype='sobel', show=False)

                                # Calculate the hash of the image
                                image_hash = calculate_image_hash(image)
                                # Check if the image hash is already in the set
                                if image_hash not in unique_image_hashes:
                                    hash_im_count +=1
                                    unique_image_hashes.add(image_hash)
                                    unique_images[image_hash] = [{
                                        'n': n,
                                        'mu_x': mu_x,
                                        'mu_y': mu_y,
                                        'sigma_x': sigma_x,
                                        'sigma_y': sigma_y,
                                        'theta':theta,
                                        'dx': dx,
                                        'dy': dy,
                                    }]

                                    # Save parameters in the parameters_matrix
                                    gaussian_parameters_matrix[mu_x_idx, mu_y_idx, sigma_x_idx, sigma_y_idx, theta_idx, dx_idx, dy_idx] = {
                                    'mu_x': round(mu_x, 2), 'mu_y': round(mu_y, 2), 'sigma_x': round(sigma_x, 2),'sigma_y':round(sigma_y,2), 'theta': round(theta, 2),'dx': dx,'dy': dy}
                                    # Save the image (modify this part as needed)
                                    image_name = f"{mu_x_idx:03d}{mu_y_idx:03d}{sigma_x_idx:03d}{sigma_y_idx:03d}{theta_idx:03d}{dx_idx:03d}{dy_idx:03d}.jpg"
                                    image_path = os.path.join(folder_name, image_name)
                                    image.save(image_path)
                                else:
                                    unique_images[image_hash].append({
                                        'n': n,
                                        'mu_x': mu_x,
                                        'mu_y': mu_y,
                                        'sigma_x': sigma_x,
                                        'sigma_y': sigma_y,
                                        'theta': theta,
                                        'dx': dx,
                                        'dy': dy,
                                    })
                                    continue  # Skip if image is a duplicate


    # Save the matrices
    print(hash_im_count)
    np.save(os.path.join(save_dir, f"parameters_matrix_{i}.npy"), parameters_matrix)
    np.save(os.path.join(save_dir, f"gaussian_parameters_matrix_{i}.npy"), gaussian_parameters_matrix)
    np.save(os.path.join(save_dir, f"unique_images_hash_{i}.npy"), unique_images)
    print("Number of unique images saved:", len(unique_image_hashes))
    with open(os.path.join(save_dir, f"final_unique_images_{i}.txt"), "w") as file:
        for key, params in unique_images.items():
            file.write(f"Key: {key}:\nParameters:  {params}\n\n")
    print("Hash file saved",i)


