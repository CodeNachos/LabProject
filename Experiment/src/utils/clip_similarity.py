from utils.image_processing import is_grayscale
import torch
import open_clip
import cv2
import numpy as np
from torchvision import transforms


# Load OpenCLIP model
model, _, preprocess = open_clip.create_model_and_transforms("ViT-H/14", pretrained="laion2b_s32b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-H-14")
#model, _, preprocess = open_clip.create_model_and_transforms("ViT-B/16", pretrained="openai")
#tokenizer = open_clip.get_tokenizer('ViT-B-16')

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def preprocess_image(image:np.array):
    """
    Loads and preprocesses an image to be used with OpenFLIP
    
    NOTE: This function expects the input image to be in Grayscale or BGR format
    as default using cv2.
    """
    if image is None:
        raise ValueError("Null reference to image.")
    
    # convert image to RGB
    if is_grayscale(image): image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) 
    else: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # resize to match OpenCLIP input size
    image = cv2.resize(image, (224, 224)) 

    # convert image to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])  # OpenCLIP normalization
    ])

    image = transform(image).unsqueeze(0).to(device)
    return image

def get_image_embedding(image):
    """
    Extracts image embedding using OpenCLIP.
    """
    if image is None:
        raise ValueError("Null reference to image.")
    
    image_tensor = preprocess_image(image)
    
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)

    return embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize

def get_text_embedding(text):
    """
    Encodes text into an embedding using OpenCLIP.
    """
    if text is None:
        raise ValueError("Null reference to text.")
    
    tokens = open_clip.tokenize([text]).to(device)  # Correct way to tokenize
    with torch.no_grad():
        embedding = model.encode_text(tokens)
    
    return embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize

def cosine_similarity(embedding1, embedding2):
    """
    Computes cosine similarity between two embeddings.
    """
    if embedding1 is None or embedding2 is None:
        raise ValueError("Null reference to embedding.")
    
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