from image_processing import *
import cv2

def main():
    image = read_image("res/test_image_2.JPG", IOFlags.GRAYSCALE)
    image = simple_thresholding(image, 127)
    display_image(image)


if __name__ == "__main__":
    main()
    