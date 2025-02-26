# IMPORTS
import os
import cv2
import zipfile
import logging
import numpy as np
from enum import IntFlag

# CONFIG =======================================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("[%(levelname)s] %(module)s - %(funcName)s: %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# CONSTANTS ====================================================================

class IOFlags(IntFlag):
    GRAYSCALE = cv2.IMREAD_GRAYSCALE
    COLOR = cv2.IMREAD_COLOR # BGR format

class ThresholdType(IntFlag):
    BINARY = cv2.THRESH_BINARY
    BINARY_INV = cv2.THRESH_BINARY_INV
    TRUNC = cv2.THRESH_TRUNC
    TOZERO = cv2.THRESH_TOZERO
    TOZERO_INV = cv2.THRESH_TOZERO_INV

class ThresholdMode(IntFlag):
    ADAPT_MEAN = cv2.ADAPTIVE_THRESH_MEAN_C
    ADPT_GAUSS = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

# UTILITY/HELPER FUNCTIONS =====================================================

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


def read_image(path, flag:IOFlags=IOFlags.COLOR) -> np.ndarray:
    """
    Reads an image from a given file path.

    Args:
        path (str): The file path to the image.
        flag (IOFlags, optional): The flag specifying how the image should be 
            read. Defaults to IOFlags.COLOR.

    Returns:
        np.ndarray: The loaded image as a NumPy array.

    Raises:
        ValueError: If the path is None.
        IOError: If the image cannot be read.
    """
    if path is None:
        raise ValueError("None reference to image path.")
    
    image = cv2.imread(path, flag)
    if image is None:
        raise IOError("Image not found or path is incorrect.")
    
    return image


def save_image(path: str, image: np.ndarray, params: list = None) -> None:
    """
    Saves an image to a given file path.

    Args:
        path (str): The file path where the image should be saved.
        image (np.ndarray): The image to be saved as a NumPy array.
        params (list, optional): Additional cv2 encoding parameters. 
            Defaults to None.

    Raises:
        ValueError: If the path or image is None.
        IOError: If the image cannot be saved.
    """
    if path is None:
        raise ValueError("None reference to image path.")
    
    if image is None:
        raise ValueError("None reference to image data.")
    
    if not cv2.imwrite(path, image, params or []):
        raise IOError("Failed to save image.")


def display_image(image:np.array, label:str="image", 
                  width:int=None, height:int=None
) -> None:
    """
    Displays an image with an adjustable window size.

    Args:
        image (np.ndarray): The image to be displayed.
        label (str, optional): The label for the display window. 
            Defaults to "image".
        width (Optional[int], optional): The desired width of the display window. 
            If None, it defaults to the image width or 1000 pixels. 
            Defaults to None.
        height (Optional[int], optional): The desired height of the display window. 
            If None, it defaults to the image height or 500 pixels.
            Defaults to None.

    Raises:
        ValueError: If the provided image is None.
    """
    if image is None:
        raise ValueError("Null reference to image.")
    
    # default window size
    dfwidth = 1000
    dfheight = 500

    # window size in function of image size
    imheight, imwidth, *_ = image.shape
    if width is None and imwidth < dfwidth:width = imwidth
    else: width = dfwidth
    if height is None and imheight < dfheight: height = imheight
    else: height = dfheight

    cv2.namedWindow(label, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(label, width, height)
    cv2.imshow(label, image)

    # wait until a key is pressed or the window is closed
    while True:
        if cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) < 1:
            break  # window closed by user
        if cv2.waitKey(1) & 0xFF == ord('q'):  # close with 'q'
            break
    cv2.destroyAllWindows()


def is_grayscale(image:np.array) -> bool: 
    """
    Checks if an image is in grayscale format.

    Args:
        image (np.ndarray): The input image.

    Returns:
        bool: True if the image is grayscale, False otherwise.
    """
    return len(image.shape) == 2 or image.shape[2] == 1


def ensure_grayscale(image):
    """
    Converts an image to grayscale if it is not already.

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The grayscale image.
    """
    if len(image.shape) == 3:  # check for 3 channels (BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # already grayscale

# IMAGE PROCESSING FUNCTIONS ===================================================

def dog(image:np.array, k:int=1.6, sigma:int=0.5, 
        gamma:int=1, ksize:tuple=(0,0)
) -> np.array:
    """
    Applies the Difference of Gaussians (DoG) filter to an image.

    Implementation inspired by 'heitorrapela' at:
    > https://github.com/heitorrapela/xdog/tree/master

    Args:
        image (np.ndarray): The input image.
        k (float, optional): The scaling factor for the second Gaussian blur. 
            Defaults to 1.6.
        sigma (float, optional): The standard deviation for the first Gaussian 
            blur. Defaults to 0.5.
        gamma (float, optional): The weighting factor for the second blurred 
            image. Defaults to 1.
        ksize (tuple, optional): The kernel size for the Gaussian blurs. 
            Defaults to (0, 0).

    Returns:
        np.ndarray: The resulting image after applying the DoG filter.

    Raises:
        ValueError: If the input image is None.
    """
    if image is None:
        raise ValueError("Null reference to image.")

    imgauss1 = cv2.GaussianBlur(image, ksize, sigma)
    imgauss2 = cv2.GaussianBlur(image, ksize, k*sigma)
    return (imgauss1 - gamma*imgauss2)


def xdog(image:np.array, sigma=0.5,
         k=200, gamma=0.98,epsilon=0.1,
         phi=10, intensity=1.0
) -> np.array:
    """
    Applies the Extended Difference of Gaussians (XDoG) filter to an image.

    Inspired by article 'XDoG: An eXtended difference-of-Gaussians compendium
    including advanced image stylization' at:
    > https://users.cs.northwestern.edu/~sco590/winnemoeller-cag2012.pdf

    Implementation inspired by 'heitorrapela' at:
    > https://github.com/heitorrapela/xdog/tree/master
    
    Args:
        image (np.ndarray): The input image.
        sigma (float, optional): The standard deviation for the first Gaussian 
            blur. Defaults to 0.5.
        k (float, optional): The scaling factor for the second Gaussian blur. 
            Defaults to 200.
        gamma (float, optional): The weighting factor for the second blurred 
            image. Defaults to 0.98.
        epsilon (float, optional): The threshold parameter. Defaults to 0.1.
        phi (float, optional): The parameter controlling sharpness. 
            Defaults to 10.
        intensity (float, optional): The intensity of the final image. 
            Defaults to 1.0.

    Returns:
        np.ndarray: The resulting XDoG filtered image.

    Raises:
        ValueError: If the input image is None.
    """
    if image is None:
        raise ValueError("None reference to image")
    
    # normalized DoG image
    imdog = dog(image, sigma=sigma, k=k, gamma=gamma)/255
    # non-linaer sharpening function
    imxdog = np.where(
        imdog >= epsilon, 1.0, 
        1.0 + np.tanh(phi * (imdog - epsilon))
    )
    # intensity control
    imxdog = np.clip(imxdog * intensity, 0, 1)
    # convert range to [0,255]
    imxdog = (imxdog * 255).astype(np.uint8)
    return imxdog


def simple_thresholding(
        image:np.array, threshold_value:int, max_value:int=255,
        type:ThresholdType=ThresholdType.BINARY
) -> np.array:
    """
    Applies simple thresholding to an image.

    Args:
        image (np.ndarray): The input image.
        threshold_value (int): The threshold value.
        max_value (int, optional): The maximum value to use for thresholding. 
            Defaults to 255.
        type (ThresholdType, optional): The thresholding type. 
            Defaults to ThresholdType.BINARY.

    Returns:
        np.ndarray: The thresholded image.

    Raises:
        ValueError: If the input image is None.
    """
    if image is None:
        raise ValueError("None reference to image")
    if not is_grayscale(image):
        logger.info("image is not grayscale, forcing convertion.")
        image = ensure_grayscale(image)

    return cv2.threshold(image, threshold_value, max_value, type)[1]


def otsu_thresholding(image:np.array, max_value:int=255,
        type:ThresholdType=ThresholdType.BINARY
) -> np.array:
    """
    Applies Otsu's thresholding method to an image.

    Args:
        image (np.ndarray): The input image.
        max_value (int, optional): The maximum value to use for thresholding. 
            Defaults to 255.
        type (ThresholdType, optional): The thresholding type. 
            Defaults to ThresholdType.BINARY.

    Returns:
        np.ndarray: The thresholded image.

    Raises:
        ValueError: If the input image is None.
    """
    if image is None:
        raise ValueError("None reference to image")
    if not is_grayscale(image):
        logger.info("image is not grayscale, forcing convertion.")
        image = ensure_grayscale(image)

    return cv2.threshold(image, 0, max_value, type + cv2.THRESH_OTSU)[1]


def adaptative_thresholding(
        image:np.array, mode:ThresholdMode, size:int, c:float,
        max_value:int=255, inverse:bool=False
) -> np.array:
    """
    Applies adaptative thresholding method to an image.

    Args:
        image (np.ndarray): The input image.
        mode (ThresholdMode): The thresholding mode (eg. MEAN or GAUSSIAN).
        size (int): The size of the neighborhood area used for thresholding 
            (must be an odd number greater than 1).
        c (float): A constant subtracted from the mean or weighted sum of the 
            neighborhood pixels.
        max_value (int, optional): The maximum pixel value assigned to the 
            thresholded pixels. Defaults to 255.
        inverse (bool, optional): Whether to apply inverse thresholding. 
            Defaults to False.


    Returns:
        np.ndarray: The thresholded image.

    Raises:
        ValueError: If the input image is None.
    """
    if image is None:
        raise ValueError("None reference to image")
    if not is_grayscale(image):
        logger.info("image is not grayscale, forcing convertion.")
        image = ensure_grayscale(image)

    return cv2.adaptiveThreshold(image, max_value, mode,
                                 cv2.THRESH_BINARY_INV if inverse 
                                 else cv2.THRESH_BINARY, 
                                 size, c
                                )


def mooney_filter(image:np.array, sigma:int, max_threshold:int=255, 
                  threshold_type:ThresholdType=ThresholdType.BINARY
) -> np.array:
    """Transforms a source image into a Mooney image using Gaussian blur and 
    Otsu's thresholding.

    This function applies a Gaussian blur to the image followed by Otsu's 
    thresholding technique. Otsu's thresholding method was chosen based on 
    research by Thomas S.A. Wallis et al. in their work: "A psychophysical 
    evaluation of techniques for Mooney image generation" available at:
    > https://arxiv.org/pdf/2403.11867

    Args:
        image (np.ndarray): The input image.
        sigma (int): The standard deviation for the Gaussian blur.
        max_threshold (int, optional): The maximum pixel value to use for thresholding. Defaults to 255.
        threshold_type (int, optional): The thresholding type (e.g., cv2.THRESH_BINARY). Defaults to cv2.THRESH_BINARY.

    Returns:
        np.ndarray: The processed Mooney image.

    Raises:
        ValueError: If the input image is None.
    """
    if image is None:
        raise ValueError("None reference to image")
    if not is_grayscale(image):
        logger.info("image is not grayscale, forcing convertion.")
        image = ensure_grayscale(image)
    
    # apply gaussian blur
    blurred_image = cv2.GaussianBlur(image, (0,0), sigma)
    
    # apply otsu thresholding and return result
    return otsu_thresholding(blurred_image, max_threshold, threshold_type)


def pixelate_image(image:np.array, pixel_size:int=10):
    """
    Applies pixelation effect to an image.

    Args:
        image (np.ndarray): The input image.
        pixel_size (int, optional): The size of the pixel blocks. Defaults to 10.

    Returns:
        np.ndarray: The pixelated image.

    Raises:
        ValueError: If the input image is None.
    """
    if image is None:
        raise ValueError("None reference to image")
    
    # get original dimensions
    height, width = image.shape[:2]
    
    # resize the image to a smaller size
    small_image = cv2.resize(image, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
    
    # scale it back to the original size using nearest neighbor interpolation
    return cv2.resize(small_image, (width, height), interpolation=cv2.INTER_NEAREST)


def bandpass_filter(image:np.array, low_cutoff:int=30, high_cutoff:int=100) -> np.array:
    """
    Applies a band-pass filter to an image using the Fourier Transform.

    This function applies a band-pass filter by removing both high and low 
    frequencies, retaining only those in the specified range.

    NOTE: Even tough frequency domain filtering is not restricted to grayscale
    images this function can only handle grayscale images and will force a 
    conversion if a colored image is given as input.
    
    Args:
        image (np.ndarray): The input grayscale image.
        low_cutoff (int, optional): The lower frequency cutoff. Defaults to 30.
        high_cutoff (int, optional): The higher frequency cutoff. Defaults to 100.

    Returns:
        np.ndarray: The filtered image.

    Raises:
        ValueError: If the input image is None.
    """
    if image is None:
        raise ValueError("None reference to image")
    if not is_grayscale(image):
        logger.info("image is not grayscale, forcing convertion.")
        image = ensure_grayscale(image)
    
    # get image dimensions
    rows, cols = image.shape
    crow, ccol = rows // 2 , cols // 2  # image center
    
    # compute fourier transform
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    
    # create band-pass mask
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.circle(mask, (ccol, crow), high_cutoff, 1, thickness=-1)  # high-pass cutoff
    cv2.circle(mask, (ccol, crow), low_cutoff, 0, thickness=-1)   # low-pass cutoff
    
    # apply the mask to the shifted DFT
    filtered_dft = dft_shift * mask
    
    # inverse transform to bring back to spatial domain
    inverse_dft = np.fft.ifftshift(filtered_dft)
    imfiltered = np.fft.ifft2(inverse_dft)
    imfiltered = np.abs(imfiltered)
    
    # return image normalized for display
    return cv2.normalize(imfiltered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    

def lowpass_filter(image:np.array, cutoff:int=30) -> np.array:
    """
    Applies a low-pass filter to an image using the Fourier Transform.

    This function removes high-frequency components, allowing only low 
    frequencies to pass.

    NOTE: Even tough frequency domain filtering is not restricted to grayscale
    images this function can only handle grayscale images and will force a 
    conversion if a colored image is given as input.
    
    Args:
        image (np.ndarray): The input grayscale image.
        cutoff (int, optional): The frequency cutoff. Defaults to 30.

    Returns:
        np.ndarray: The filtered image.

    Raises:
        ValueError: If the input image is None.
    """
    if image is None:
        raise ValueError("None reference to image")
    if not is_grayscale(image):
        logger.info("image is not grayscale, forcing convertion.")
        image = ensure_grayscale(image)

    # apply fourier transform
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    
    # create low-pass mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), cutoff, 1, thickness=-1)
    
    # apply mask
    filtered_dft = dft_shift * mask

    # apply inverse fourier transform
    inverse_dft = np.fft.ifftshift(filtered_dft)
    imfiltered = np.fft.ifft2(inverse_dft)
    imfiltered = np.abs(imfiltered)

    # return normalized image 
    return cv2.normalize(imfiltered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    

def highpass_filter(image:np.array, cutoff:int=30) -> np.array:
    """
    Applies a high-pass filter to an image using the Fourier Transform.

    This function removes low-frequency components, retaining only high 
    frequencies.

    NOTE: Even tough frequency domain filtering is not restricted to grayscale
    images this function can only handle grayscale images and will force a 
    conversion if a colored image is given as input.
    
    Args:
        image (np.ndarray): The input grayscale image.
        cutoff (int, optional): The frequency cutoff. Defaults to 30.

    Returns:
        np.ndarray: The filtered image.

    Raises:
        ValueError: If the input image is None.
    """
    if image is None:
        raise ValueError("None reference to image")
    if not is_grayscale(image):
        logger.info("image is not grayscale, forcing convertion.")
        image = ensure_grayscale(image)

    # Ffourier transform
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    
    # create high-pass mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), cutoff, 0, thickness=-1)
    
    # apply mask
    filtered_dft = dft_shift * mask

    # apply inverse fourier transform
    inverse_dft = np.fft.ifftshift(filtered_dft)
    imfiltered = np.fft.ifft2(inverse_dft)
    imfiltered = np.abs(imfiltered)

    # return normalized image
    return cv2.normalize(imfiltered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
