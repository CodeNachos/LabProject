# Image Processing Module

## Overview
This module provides a set of image processing functions, including image reading, display, grayscale conversion, edge detection, and frequency-based filtering. It is built using OpenCV and NumPy, supporting some common image transformation techniques.

---

## Dependencies
This module requires the following libraries:
- `opencv-python` (`cv2`)
- `numpy`
- `zipfile`
- `logging`
- `os`
- `enum`

---

## Configuration
### Logger Setup
A logger is configured to output messages for debugging and information logging.

---

## Constants
### IOFlags
Defines flags for image reading:
- `UNCHANGED`
- `GRAYSCALE`
- `COLOR`

### ThresholdType
Defines thresholding types:
- `BINARY`
- `BINARY_INV`
- `TRUNC`
- `TOZERO`
- `TOZERO_INV`

### ThresholdMode
Defines adaptive thresholding modes:
- `ADAPT_MEAN`
- `ADPT_GAUSS`

---

## Utility Functions

### `extract_images_from_zip(zip_path, paths_file, output_folder)`
Extracts images from a zip archive based on a provided list of file paths.

**Parameters:**
- `zip_path (str)`: Path to the zip file.
- `paths_file (str)`: Path to a text file containing image paths to extract.
- `output_folder (str)`: Destination folder for extracted images.

---

## Image I/O Functions

### `read_image(path, flag=IOFlags.UNCHANGED) -> np.ndarray`
Reads an image from a given path.

**Parameters:**
- `path (str)`: File path of the image.
- `flag (IOFlags)`: Flag for reading mode.

**Returns:**
- `np.ndarray`: Loaded image.

**Raises:**
- `ValueError`: If path is `None`.
- `IOError`: If image cannot be read.

### `display_image(image, label="image", width=None, height=None) -> None`
Displays an image in an OpenCV window.

**Parameters:**
- `image (np.ndarray)`: Image to display.
- `label (str)`: Window label.
- `width (int)`: Display width.
- `height (int)`: Display height.

**Raises:**
- `ValueError`: If image is `None`.

---

## Image Processing Functions

### `is_grayscale(image) -> bool`
Checks if an image is in grayscale format.

**Parameters:**
- `image (np.ndarray)`: Input image.

**Returns:**
- `bool`: `True` if grayscale, `False` otherwise.

### `ensure_grayscale(image) -> np.ndarray`
Converts an image to grayscale if necessary.

**Parameters:**
- `image (np.ndarray)`: Input image.

**Returns:**
- `np.ndarray`: Grayscale image.

---

## Edge Detection Filters

### `dog(image, k=1.6, sigma=0.5, gamma=1, ksize=(0,0)) -> np.ndarray`
Applies the Difference of Gaussians (DoG) filter.

**Parameters:**
- `image (np.ndarray)`: Input image.
- `k (float)`: Scale factor for second blur.
- `sigma (float)`: Standard deviation for first blur.
- `gamma (float)`: Weighting factor for second blur.
- `ksize (tuple)`: Kernel size.

**Returns:**
- `np.ndarray`: Processed image.

**Raises:**
- `ValueError`: If image is `None`.

### `xdog(image, sigma=0.5, k=200, gamma=0.98, epsilon=0.1, phi=10, intensity=1.0) -> np.ndarray`
Applies the Extended Difference of Gaussians (XDoG) filter.

**Parameters:**
- `image (np.ndarray)`: Input image.
- `sigma (float)`: Standard deviation for blur.
- `k (float)`: Scaling factor for second blur.
- `gamma (float)`: Weighting factor.
- `epsilon (float)`: Threshold parameter.
- `phi (float)`: Sharpness control.
- `intensity (float)`: Output intensity.

**Returns:**
- `np.ndarray`: Processed image.

**Raises:**
- `ValueError`: If image is `None`.

---

## Thresholding Functions

### `simple_thresholding(image, threshold_value, max_value=255, type=ThresholdType.BINARY) -> np.ndarray`
Applies simple thresholding.

**Parameters:**
- `image (np.ndarray)`: Input image.
- `threshold_value (int)`: Threshold value.
- `max_value (int)`: Maximum output value.
- `type (ThresholdType)`: Thresholding type.

**Returns:**
- `np.ndarray`: Thresholded image.

**Raises:**
- `ValueError`: If image is `None`.

### `otsu_thresholding(image, max_value=255, type=ThresholdType.BINARY) -> np.ndarray`
Applies Otsu's thresholding.

**Parameters:**
- `image (np.ndarray)`: Input image.
- `max_value (int)`: Maximum output value.
- `type (ThresholdType)`: Thresholding type.

**Returns:**
- `np.ndarray`: Thresholded image.

**Raises:**
- `ValueError`: If image is `None`.

### `adaptative_thresholding(image, mode, size, c, max_value=255, inverse=False) -> np.ndarray`
Applies adaptive thresholding.

**Parameters:**
- `image (np.ndarray)`: Input image.
- `mode (ThresholdMode)`: Adaptive thresholding mode.
- `size (int)`: Block size.
- `c (float)`: Constant subtracted from mean.
- `max_value (int)`: Maximum output value.
- `inverse (bool)`: Apply inverse thresholding.

**Returns:**
- `np.ndarray`: Thresholded image.

**Raises:**
- `ValueError`: If image is `None`.

### `mooney_filter(image, sigma, max_threshold=255, threshold_type=ThresholdType.BINARY) -> np.ndarray`
Applies Gaussian blur followed by Otsu’s thresholding.

**Parameters:**
- `image (np.ndarray)`: Input image.
- `sigma (int)`: Gaussian blur standard deviation.
- `max_threshold (int)`: Maximum output value.
- `threshold_type (ThresholdType)`: Thresholding type.

**Returns:**
- `np.ndarray`: Mooney image.

**Raises:**
- `ValueError`: If image is `None`.

---

## Frequency Filters

### `bandpass_filter(image, low_cutoff=30, high_cutoff=100) -> np.ndarray`
Applies a band-pass filter using Fourier Transform.

### `lowpass_filter(image, cutoff=30) -> np.ndarray`
Applies a low-pass filter using Fourier Transform.

### `highpass_filter(image, cutoff=30) -> np.ndarray`
Applies a high-pass filter using Fourier Transform.

**Parameters (for all three filters):**
- `image (np.ndarray)`: Input image.
- `cutoff (int)`: Frequency cutoff.

**Returns:**
- `np.ndarray`: Filtered image.

**Raises:**
- `ValueError`: If image is `None`.

---

## Miscellaneous Functions

### `pixelate_image(image, pixel_size=10) -> np.ndarray`
Applies a pixelation effect.

**Parameters:**
- `image (np.ndarray)`: Input image.
- `pixel_size (int)`: Pixel block size.

**Returns:**
- `np.ndarray`: Pixelated image.

**Raises:**
- `ValueError`: If image is `None`.

---

## Usage Example

```python
import cv2
from image_processing import read_image, display_image, dog

image = read_image("example.jpg")
filtered_image = dog(image)

display_image(filtered_image)
