from image_processing import *
import cv2
from clip_test import *

def main():
    # Example usage
    image1 = read_image("res/test_image_house.jpg", flag=IOFlags.GRAYSCALE) 
    image2 = read_image("res/test_image_cat.jpg", flag=IOFlags.GRAYSCALE)

    imfiltered1 = xdog(image1, sigma=0.3, k=100, gamma=.8, epsilon=0.3, phi=3)
    imfiltered1 = otsu_thresholding(imfiltered1)

    display_image(imfiltered1)
    
    descriptive_text = """
A two-story suburban house with a light gray exterior, 
dark brown roof, and a wooden front door.
"""

    moderate_text = """A two-story house."""

    simple_text = "House."

    similarity1 = compare_images(image1, imfiltered1)
    similarity2 = compare_images(image1, image2)
    similarity3 = compare_image_text(
        image1, 
        descriptive_text
    )
    similarity4 = compare_image_text(
        image1,
        moderate_text
    )
    similarity5 = compare_image_text(
        image1,
        simple_text
    )
    print(f"Image1 vs image1 filtered similarity : {similarity1} + {get_ssim(image1, imfiltered1)}")
    print(f"Image1 vs image2 similarity          : {similarity2} + {get_ssim(image1, image2)}")
    print(f"Image1 vs detailed text similarity   : {similarity3}")
    print(f"Image1 vs moderate text similarity   : {similarity4}")
    print(f"Image1 vs label text similarity      : {similarity5}")

    
    
if __name__ == "__main__":
    main()