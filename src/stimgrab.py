import os
import logging
import zipfile
import numpy as np
from enum import IntEnum
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

import utils.image_processing as imp

# CONSTANTS ====================================================================


class Modalities(IntEnum):
    LOW = 1
    MEDIUMLOW = 2
    MEDIUM = 3
    HIGH = 4

    def __str__(self):
        if self == self.LOW:
            return "low"
        elif self == self.MEDIUMLOW:
            return "medium-low"
        elif self == self.MEDIUM:
            return "medium"
        elif self == self.HIGH:
            return "high"

IMDIM = (224, 224)

_DATASET_URL = 'http://olivalab.mit.edu/MM/downloads/Scenes.zip'

_SRC_DIR = Path(__file__).resolve().parent
_CATEGORIES_PATH = (_SRC_DIR / "../res/expdata/categories.txt").resolve()
_DATASET_PATH = (_SRC_DIR / "../res/datasets/mit_stimuli_scenes.zip").resolve()
_STIMULI_PATH = (_SRC_DIR / "../res/stimuli").resolve()
_ORIGIN_PATH = _STIMULI_PATH / "original"
_GRAYSCALE_PATH = _STIMULI_PATH / "grayscale"
_HIMOD_PATH = _STIMULI_PATH / "trialready/high"
_MIDMOD_PATH = _STIMULI_PATH / "trialready/medium"
_MIDLOWMOD_PATH = _STIMULI_PATH / "trialready/mediumlow"
_LOWMOD_PATH = _STIMULI_PATH / "trialready/low"

_MODELS_PATH = (_SRC_DIR / "../res/aimodels").resolve()

# CONFIG =======================================================================
_default_logger_level = logging.INFO 

logger = logging.getLogger(__name__)
logger.setLevel(_default_logger_level)
logger.propagate = False

console_handler = logging.StreamHandler()
console_handler.setLevel(_default_logger_level)

formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


# UTILITY ======================================================================

is_image_modality = lambda m : m in [Modalities.HIGH, Modalities.MEDIUM]
is_text_modality  = lambda m : m in [Modalities.MEDIUMLOW, Modalities.LOW]

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


# Init ML models

def _init_blip():
    logger.info("[INFO]: Loading BLIP model...")

    import torch
    from transformers import AutoProcessor, Blip2ForConditionalGeneration
    model_name = "Salesforce/blip2-flan-t5-xl"
    
    # load processor and model
    if not hasattr(text_modality_transform, "PROCESSOR"):
        text_modality_transform.PROCESSOR = AutoProcessor.from_pretrained(
            model_name,
            use_fast=True,
            cache_dir=_MODELS_PATH
        )
    if not hasattr(text_modality_transform, "MODEL"):
        text_modality_transform.MODEL = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            device_map = {"": "cpu"},
            torch_dtype = torch.float32,
            low_cpu_mem_usage = True,
            cache_dir=_MODELS_PATH
        )

def _init_opus():
    logger.info("[INFO]: Loading OPUS model...")
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    # load the model and tokenizer once
    model_name = "Helsinki-NLP/opus-mt-en-fr"

    if not hasattr(_translate_text_modalities, "tokenizer"):
        _translate_text_modalities.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=_MODELS_PATH
        )
    if not hasattr(_translate_text_modalities, "model"):
        _translate_text_modalities.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=_MODELS_PATH
        )
    
def _init_aimodels():
    logger.info("[INFO]: Loading AI models...")
    import torch
    from transformers import AutoProcessor, Blip2ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
    _init_blip()
    _init_opus()

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

# Modality transforms
# Image modalities 

def high_modality_transform(image:np.ndarray) -> np.ndarray:
    """
    Applies high congruence processing to the image.

    This method applies a low-pass filter to the image to modify its congruence 
    characteristics.

    Args:
        img (numpy.ndarray): The image to process.
    """
    return imp.lowpass_filter(image, 50)

def medium_modality_transform(image:np.ndarray) -> np.ndarray:
    """
    Applies medium congruence processing to the image.

    This method applies XDoG (Extended Difference of Gaussian) and Otsu 
    thresholding to the image.

    Args:
        img (numpy.ndarray): The image to process.
    """
    hc_img = imp.xdog(image, sigma=0.5, k=100, gamma=.7, epsilon=0.3, phi=1)
    return imp.otsu_thresholding(hc_img)

def image_modality_transform(image:np.ndarray, image_path:Path, modality:Modalities):
    if not is_image_modality(modality):
        raise ValueError("Modality is not an image modality.")

    output_image = None

    if modality == Modalities.HIGH:
        output_image = high_modality_transform(image)
        dest = Path(f"{_HIMOD_PATH}{image_path.as_posix()[len(_ORIGIN_PATH.as_posix()):]}")
    elif modality == Modalities.MEDIUM:
        output_image = medium_modality_transform(image)
        dest = Path(f"{_MIDMOD_PATH}{image_path.as_posix()[len(_ORIGIN_PATH.as_posix()):]}")
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    imp.save_image(dest, output_image)

# Text modalities

def _translate_text_modalities():
    # load the model and tokenizer once
    #if not hasattr(_translate_text_modalities, "model"):
    #    _init_opus
    
    #import torch
    
    #tokenizer = _translate_text_modalities.tokenizer
    #model = _translate_text_modalities.model

    # find all .txt files that are not already translated
    for txt_path in _STIMULI_PATH.rglob("*_en.txt"):
        if txt_path.stem.endswith("_fr"):
            continue  # skip already translated files just in case

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue  # skip empty files

        translation = "Missing stimulus description"
        # translate
        #inputs = tokenizer(text, return_tensors="pt")
        #translated_ids = model.generate(**inputs, max_length=100)
        #translation = tokenizer.batch_decode(
        #    translated_ids, 
        #    skip_special_tokens=True
        #)[0].strip()

        # save to a new file ending with _fr.txt
        translated_path = txt_path.with_name(
            txt_path.stem.replace("_en", "_fr") + ".txt"
        )
        with open(translated_path, "w", encoding="utf-8") as f:
            f.write(translation)

        logger.debug(f">> Translated: {txt_path.name} -> {translated_path.name}")

def text_modality_transform(image:np.ndarray, image_path:Path, modality:Modalities):
    if not is_text_modality(modality):
        raise ValueError("Modality is not a text modality.")

    MEDIUMLOW_PROMPT = ("Context: This is a scene description task."
        "Question: In one or two sentences, describe the structure in this image"
        "— its shapes, forms, and main architectural components. Answer:" 
    )
    LOW_PROMPT = ("Context: This is a scene description task."
        "Label the category of the main object or structure in this image? Answer:"
    )

    # load processor and model
    #if not hasattr(text_modality_transform, "MODEL"):
    #    _init_blip
    
    #import torch
    #device = torch.device("cpu")

    # Prepare inputs
    relative_path = image_path.as_posix()[len(_ORIGIN_PATH.as_posix()):]
    
    import cv2
    blip_image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

    if modality == Modalities.MEDIUMLOW:
        #inputs = text_modality_transform.PROCESSOR(
        #    images=blip_image, text=MEDIUMLOW_PROMPT, 
        #    return_tensors="pt"
        #).to(device, dtype=torch.float16)
        dest = Path(f"{_MIDLOWMOD_PATH}{relative_path}")
        dest = dest.with_name(dest.stem + "_en.txt")
    
    elif modality == Modalities.LOW:
        #inputs = text_modality_transform.PROCESSOR(
        #    images=blip_image, text=LOW_PROMPT, 
        #    return_tensors="pt"
        #).to(device, dtype=torch.float16)
        dest = Path(f"{_LOWMOD_PATH}{relative_path}")
        dest = dest.with_name(dest.stem + "_en.txt")

    description = "Missing stimulus description"

    # generate output
    #generated_ids = text_modality_transform.MODEL.generate(**inputs, max_new_tokens=30)
    #description = text_modality_transform.PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    logger.debug(">>",description)

    # save the description
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(description)

# General image processing

def process_stimuli():
    """
    Processes all images by ensuring grayscale and applying congruence 
    methods.

    This function will convert each image to grayscale, apply modalities 
    transform methods, saving each processed image to the appropriate directory.
    """
    #_init_aimodels()

    logger.info("")
    
    for img_path in _ORIGIN_PATH.rglob("*.jpg"):
        logger.info(f"Treating \"{img_path.as_posix()[len(_ORIGIN_PATH.as_posix()):]}\"")
        img = imp.read_image(img_path, flag=imp.IOFlags.GRAYSCALE, dim=IMDIM)

        logger.debug("-   Ensuring grayscale...")
        img = imp.ensure_grayscale(img)
        dest = Path(f"{_GRAYSCALE_PATH}{img_path.as_posix()[len(_ORIGIN_PATH.as_posix()):]}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        imp.save_image(dest, img)

        for m in list(Modalities):
            logger.debug(f"-   Converting using {m} congruence transform...")
            if is_text_modality(m): 
                text_modality_transform(img, img_path, m)
            elif is_image_modality(m): image_modality_transform(img, img_path, m)

        _translate_text_modalities()

    logger.info("\nImage treatment complete!")


# MAIN EXECUTION ===============================================================

def create_stimuli(nb_images:int=0, category_list:str=None):
    verify_dataset()
    logger.info("")
    extract_images(nb_images, category_list)
    logger.info("")
    process_stimuli()

if __name__ == "__main__":
    create_stimuli(nb_images=5)