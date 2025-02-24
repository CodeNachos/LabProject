# IMPORTS
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import zipfile
import os

__IMREAD_FLAGS = [
    cv2.IMREAD_UNCHANGED,
    cv2.IMREAD_GRAYSCALE,
    cv2.IMREAD_COLOR
]

__logging:bool=True
__debug:bool=True

# UTILITY/HELPER FUNCTIONS =====================================================

def toggle_logging():
    global __logging
    __logging = not __logging

def logging():
    return __logging

def toggle_debug():
    global __debug
    __debug = not __debug

def debug():
    return __debug


def extract_images_from_zip(zip_path, paths_file, output_folder):
    # Read image paths from the text file
    with open(paths_file, 'r') as file:
        image_paths = [line.strip() for line in file.readlines()]
    
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract specified images from the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for image_path in image_paths:
            if image_path in zip_ref.namelist():
                zip_ref.extract(image_path, output_folder)
                print(f"Extracted: {image_path}")
            else:
                print(f"Image not found in the zip: {image_path}")
    
    print("Extraction complete.")

def read_image(path, flag=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    if flag not in __IMREAD_FLAGS:
        raise ValueError("Invalid flag value provided for image reading.")
    
    image = cv2.imread(path, flag)
    if image is None:
        raise ValueError("Image not found or path is incorrect.")
    
    return image

def display_image(image, cmap=None):
    plt.figure(figsize=(10, 5))

    # Automatically handle RGB vs grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert BGR to RGB if the image has 3 channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)  # Don't use cmap for RGB images
    else:
        # For grayscale images, apply the given or default cmap
        plt.imshow(image, cmap=cmap or plt.rcParams['image.cmap'])
    plt.axis('off')
    plt.show()

# IMAGE PROCESSING FUNCTIONS ===================================================

def xdog(image, sigma=0.3, k=1.6, gamma=0.98, epsilon=-0.1, phi=10):
    if image is None:
        raise ValueError("None reference to image")
    
    # Apply two Gaussian blurs with different sigmas
    G1 = cv2.GaussianBlur(image, (0, 0), sigma)
    G2 = cv2.GaussianBlur(image, (0, 0), sigma * k)
    
    # Difference of Gaussians
    DoG = G1 - gamma * G2
    
    # Normalize DoG values between 0 and 1
    DoG = DoG / 255.0
    
    # Apply the non-linear function for edge enhancement
    XDoG = np.where(DoG < epsilon, 1.0, 1.0 + np.tanh(phi * (DoG - epsilon)))
    
    # Rescale to 0-255 for image display
    XDoG = np.uint8(XDoG * 255)
    
    return XDoG


def otsu_thresholding(image_path):
    # Load the image and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or path is incorrect.")
    
    # Apply Otsu's thresholding
    _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return otsu_thresh

def mooney_filter(image_path, blur_sigma=5, threshold_value=127):
    # Load the image and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or path is incorrect.")
    
    # Apply Gaussian blur to smooth the image
    blurred_image = cv2.GaussianBlur(image, (0, 0), blur_sigma)
    
    # Apply binary thresholding to create a high-contrast image
    _, mooney_image = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY)
    
    return mooney_image

def pixelate_image(image_path, pixel_size=10):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect.")
    
    # Get original dimensions
    height, width = image.shape[:2]
    
    # Resize the image to a smaller size
    small_image = cv2.resize(image, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
    
    # Scale it back to the original size using nearest-neighbor interpolation
    pixelated_image = cv2.resize(small_image, (width, height), interpolation=cv2.INTER_NEAREST)
    
    return pixelated_image

def band_pass_filter(image_path, low_cutoff=30, high_cutoff=100):
    # Load image and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or path is incorrect.")
    
    # Get the image dimensions
    rows, cols = image.shape
    crow, ccol = rows // 2 , cols // 2  # Center of the image
    
    # Apply Fourier Transform
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)  # Shift the zero frequency component to the center
    
    # Create a band-pass filter mask
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.circle(mask, (ccol, crow), high_cutoff, 1, thickness=-1)  # High-pass cutoff
    cv2.circle(mask, (ccol, crow), low_cutoff, 0, thickness=-1)   # Low-pass cutoff (creates a band)
    
    # Apply the mask to the shifted DFT
    filtered_dft = dft_shift * mask
    
    # Inverse Fourier Transform to bring back to spatial domain
    inverse_dft = np.fft.ifftshift(filtered_dft)
    img_back = np.fft.ifft2(inverse_dft)
    img_back = np.abs(img_back)
    
    # Normalize the image for display
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return img_back


def low_pass_filter(image_path, cutoff=30):
    # Load image and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or path is incorrect.")
    
    # Fourier Transform
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    
    # Create Low-Pass Filter mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), cutoff, 1, thickness=-1)
    
    # Apply mask and inverse DFT
    filtered_dft = dft_shift * mask
    inverse_dft = np.fft.ifftshift(filtered_dft)
    img_back = np.fft.ifft2(inverse_dft)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return img_back

def high_pass_filter(image_path, cutoff=30):
    # Load image and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or path is incorrect.")
    
    # Fourier Transform
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    
    # Create High-Pass Filter mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), cutoff, 0, thickness=-1)
    
    # Apply mask and inverse DFT
    filtered_dft = dft_shift * mask
    inverse_dft = np.fft.ifftshift(filtered_dft)
    img_back = np.fft.ifft2(inverse_dft)
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return img_back

