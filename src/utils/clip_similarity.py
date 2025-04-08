import logging
import numpy as np

_default_logger_level = logging.INFO 

logger = logging.getLogger(__name__)
logger.setLevel(_default_logger_level)
logger.propagate = False

console_handler = logging.StreamHandler()
console_handler.setLevel(_default_logger_level)

formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

_initialized = False

_model = None
_device = None

_is_initialized = lambda : _initialized

def init_clip():
    logger.info("[INFO]: Loading OpenCLIP model...")

    global torch; global transforms; global open_clip

    import torch
    import open_clip
    from torchvision import transforms

    global _model; global _device; global _initialized
    # Load OpenCLIP _model
    _model, _, _ = open_clip.create_model_and_transforms("ViT-H/14", pretrained="laion2b_s32b_b79k")

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(_device)

    _initialized = True

def _verify_clip():
    if not _is_initialized():
        raise AssertionError("OpenCLIP model is not itilialized, think about running init_clip() to initialize the necessary dependencies.")
    

def preprocess_image(image:np.array):
    """
    Loads and preprocesses an image to be used with OpenFLIP
    
    NOTE: This function expects the input image to be in Grayscale or BGR format
    as default using cv2.
    """
    if image is None:
        raise ValueError("Null reference to image.")
    
    _verify_clip()

    # convert image to RGB
    import cv2
    from utils.image_processing import is_grayscale
    
    if is_grayscale(image): image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) 
    else: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # resize to match OpenCLIP input size
    image = cv2.resize(image, (224, 224)) 

    # convert image to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])  # OpenCLIP normalization
    ])

    image = transform(image).unsqueeze(0).to(_device)
    return image

def get_image_embedding(image):
    """
    Extracts image embedding using OpenCLIP.
    """
    if image is None:
        raise ValueError("Null reference to image.")
    
    _verify_clip()

    image_tensor = preprocess_image(image)
    
    with torch.no_grad():
        embedding = _model.encode_image(image_tensor)

    return embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize

def get_text_embedding(text):
    """
    Encodes text into an embedding using OpenCLIP.
    """
    if text is None:
        raise ValueError("Null reference to text.")
    
    _verify_clip()

    tokens = open_clip.tokenize([text]).to(_device)  # Correct way to tokenize
    with torch.no_grad():
        embedding = _model.encode_text(tokens)
    
    return embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize

def cosine_similarity(embedding1, embedding2):
    """
    Computes cosine similarity between two embeddings.
    """
    if embedding1 is None or embedding2 is None:
        raise ValueError("Null reference to embedding.")
    
    import torch
    return torch.nn.functional.cosine_similarity(embedding1, embedding2).item()

def compare_images(image1, image2):
    """
    Compares two images and returns similarity score.
    """
    if image1 is None or image2 is None:
        raise ValueError("Null reference to image.")
    
    embedding1 = get_image_embedding(image1)
    embedding2 = get_image_embedding(image2)

    return cosine_similarity(embedding1, embedding2)

def compare_image_text(image, text):
    """
    Compares an image to a text description using cosine similarity.
    """
    if image is None:
        raise ValueError("Null reference to image.")
    if text is None:
        raise ValueError("Null reference to text.")
    
    image_embedding = get_image_embedding(image)
    text_embedding = get_text_embedding(text)

    return cosine_similarity(image_embedding, text_embedding)