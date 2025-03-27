import os
import zipfile
import logging
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError
import utils.image_processing as imp

# CONSTANTS ====================================================================

IMDIM = (224, 224)

_DATASET_URL = 'http://olivalab.mit.edu/MM/downloads/Scenes.zip'

_SRC_DIR = Path(__file__).resolve().parent
_CATEGORIES_PATH = (_SRC_DIR / "../res/expdata/categories.txt").resolve()
_DATASET_PATH = (_SRC_DIR / "../res/datasets/mit_stimuli_scenes.zip").resolve()
_STIMULI_PATH = (_SRC_DIR / "../res/stimuli").resolve()
_ORIGIN_PATH = _STIMULI_PATH / "original"
_GRAYSCALE_PATH = _STIMULI_PATH / "grayscale"
_HICONGR_PATH = _STIMULI_PATH / "trialready/high"
_MIDCONGR_PATH = _STIMULI_PATH / "trialready/medium"

# CONFIG =======================================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


# UTILITY ======================================================================

def _download_progress(blocknum, blocksize, totalsize):
    """
    Displays the download progress in percentage and MB.

    Args:
        blocknum (int): The current block number.
        blocksize (int): The size of each block in bytes.
        totalsize (int): The total size of the file being downloaded in bytes.
    """
    if logger.level > logging.INFO:
        return
    
    downloaded = blocknum * blocksize
    if totalsize > 0:
        percent = downloaded * 100 / totalsize
        downloaded_mb = downloaded / (1024 * 1024)
        totalsize_mb = totalsize / (1024 * 1024)
        print(f"\r\tDownloading: {percent:.2f}% ({downloaded_mb:.2f}MB/{totalsize_mb:.2f}MB)", end="")
    else:
        downloaded_mb = downloaded / (1024 * 1024)
        print(f"\rDownloading: {downloaded_mb:.2f}MB", end="")


# DATSET =======================================================================

def verify_dataset():
    """
    Downloads the dataset from the specified URL if it does not already exist.

    Checks if the dataset is already present at the specified path. If not, it 
    downloads the dataset and saves it at the specified location.

    Raises:
        HTTPError: If an HTTP error occurs during the download.
        URLError: If a URL error occurs during the download.
        Exception: For any other errors encountered during the download.
    """
    if not os.path.isfile(_DATASET_PATH):
        try:
            logger.info(f"[INFO] Dataset file not found! Downloading from {_DATASET_URL}...\n")
            os.makedirs(_DATASET_PATH.parent, exist_ok=True)
            urlretrieve(_DATASET_URL, _DATASET_PATH, reporthook=_download_progress)
            logger.info("\nDataset successfully downloaded!")
        except HTTPError as e:
            logger.error(f"HTTP error occurred: {e.code} - {e.reason}")
        except URLError as e:
            logger.error(f"URL error occurred: {e.reason}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
    else:
        logger.info("Dataset file found!")


def extract_images(num_images=0, category_list_file=None):
    """
    Extracts images from the dataset zip file based on specified categories.

    Args:
        num_images (int): The number of images to extract per category. If 0, 
            all images are extracted.
        category_list_file (str or None): Path to a text file containing the 
            list of categories. If None, uses the default categories file.

    Raises:
        FileNotFoundError: If the specified category file does not exist.
    """
    os.makedirs(_ORIGIN_PATH, exist_ok=True)

    if category_list_file is None:
        category_list_file = _CATEGORIES_PATH

    with open(category_list_file, 'r') as file:
        categories = [line.strip() for line in file if line.strip()]

    logger.info("Extracting images...\n")

    with zipfile.ZipFile(_DATASET_PATH, 'r') as zip_ref:
        for folder in categories:
            logger.info(f"-   Extracting {folder}...")
            matching_files = [
                f for f in zip_ref.namelist() 
                if f.startswith(f"{folder}/") and f.endswith('.jpg')
            ]

            if matching_files:
                if num_images > 0 and num_images < len(matching_files):
                    matching_files = matching_files[:num_images]

                for file in matching_files:
                    zip_ref.extract(file, _ORIGIN_PATH)
            else:
                logger.error(f"No files matched in {folder}")
        logger.info("\nExtraction complete!")


# IMG PROCESSING ===============================================================

def medium_congruence(img, img_path):
    """
    Applies medium congruence processing to the image.

    This method applies a low-pass filter to the image to modify its congruence 
    characteristics.

    Args:
        img (numpy.ndarray): The image to process.
        img_path (Path): The path to the original image for saving the processed 
            version.

    Raises:
        FileNotFoundError: If the destination directory for saving the processed 
            image does not exist.
    """
    mc_img = imp.lowpass_filter(img, 50)
    dest = Path(f"{_HICONGR_PATH}{img_path.as_posix()[len(_ORIGIN_PATH.as_posix()):]}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    imp.save_image(dest, mc_img)


def high_congruence(img, img_path):
    """
    Applies high congruence processing to the image.

    This method applies XDoG (Extended Difference of Gaussian) and Otsu t
    hresholding to the image.

    Args:
        img (numpy.ndarray): The image to process.
        img_path (Path): The path to the original image for saving the processed 
            version.

    Raises:
        FileNotFoundError: If the destination directory for saving the processed 
            image does not exist.
    """
    hc_img = imp.xdog(img, sigma=0.5, k=100, gamma=.7, epsilon=0.3, phi=1)
    hc_img = imp.otsu_thresholding(hc_img)
    dest = Path(f"{_MIDCONGR_PATH}{img_path.as_posix()[len(_ORIGIN_PATH.as_posix()):]}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    imp.save_image(dest, hc_img)


def process_stimuli():
    """
    Processes all images by ensuring grayscale and applying congruence 
    methods.

    This function will convert each image to grayscale, apply the medium 
    congruence method, and apply the high congruence method, saving each 
    processed image to the appropriate directory.

    Raises:
        FileNotFoundError: If the destination directory for saving the processed 
            images does not exist.
    """
    for img_path in _ORIGIN_PATH.rglob("*.jpg"):
        logger.info(f"Treating \"{img_path.as_posix()[len(_ORIGIN_PATH.as_posix()):]}\"")
        img = imp.read_image(img_path, flag=imp.IOFlags.GRAYSCALE, dim=IMDIM)

        logger.debug("-   Ensuring grayscale...")
        img = imp.ensure_grayscale(img)
        dest = Path(f"{_GRAYSCALE_PATH}{img_path.as_posix()[len(_ORIGIN_PATH.as_posix()):]}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        imp.save_image(dest, img)

        logger.debug("-   Converting using medium congruence method...")
        high_congruence(img, img_path)

        logger.debug("-   Converting using high congruence method...")
        medium_congruence(img, img_path)

    logger.info("\nImage treatment complete!")


# MAIN EXECUTION ===============================================================

def create_stimuli(nb_images:int=0, category_list:str=None):
    verify_dataset()
    logger.info("")
    extract_images(nb_images, category_list)
    logger.info("")
    process_stimuli()

if __name__ == "__main__":
    verify_dataset()
    logger.info("")
    extract_images(4, "test.txt")
    logger.info("")
    process_stimuli()
