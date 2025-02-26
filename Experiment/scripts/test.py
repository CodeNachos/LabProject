from image_processing import *
import cv2
from clip_test import *

def main():
    # Example usage
    image1 = read_image("res/test_image_house.jpg", flag=IOFlags.GRAYSCALE) 
    image2 = read_image("res/test_image_castle.jpg", flag=IOFlags.GRAYSCALE)

    imfiltered1 = xdog(image1, sigma=0.3, k=100, gamma=.8, epsilon=0.3, phi=3)
    imfiltered1 = otsu_thresholding(imfiltered1)

    display_image(imfiltered1)
    
    descriptive_text = """
A two-story suburban house with a light gray exterior, 
dark brown roof, and a wooden front door. A circular window sits above the 
entrance. The house has a garage on the right and a small fenced yard with 
brick pillars. A "For Sale" sign is near the driveway. The wet road and 
overcast sky suggest rainy weather, with leafless trees in the background.
"""

    moderate_text = """
A modern two-story house with a gray exterior, dark roof, and a wooden door. 
It has a garage, a small fenced yard, and a "For Sale" sign. The wet road and 
cloudy sky indicate rain.
"""

    simple_text = "Two-story house for sale on a rainy day."

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
    print(f"Image1 vs image1 filtered similarity : {similarity1}")
    print(f"Image1 vs image2 similarity          : {similarity2}")
    print(f"Image1 vs detailed text similarity   : {similarity3}")
    print(f"Image1 vs moderate text similarity   : {similarity4}")
    print(f"Image1 vs label text similarity      : {similarity5}")

    
    
if __name__ == "__main__":
    main()