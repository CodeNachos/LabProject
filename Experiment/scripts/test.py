import image_processing as impross
from cv2 import IMREAD_GRAYSCALE

def main():
    image = impross.read_image("scripts/res/test_image_1.jpeg", flag=IMREAD_GRAYSCALE)
    image = impross.xdog(image, sigma=.5, k=2, gamma=0.8, epsilon=-0.05, phi=15)
    impross.display_image(image, cmap="gray")

if __name__ == "__main__":
    main()