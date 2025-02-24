import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

IMG_RGB_PATH:str = "res/images.jpeg"

# read rgb image
img_rgb = ski.io.imread(IMG_RGB_PATH)

# 1) Display image
def display_image(img):
    plt.imshow(
        img, 
        cmap="viridis" if len(img.shape) == 3 else "gray"
        )
    plt.show()

# 2) Convert image to grayscale
img_gray = ski.color.rgb2gray(img_rgb)

# display rgb and grayscale images
#display_image(img_rgb)
#display_image(img_gray)

# FOR GRAYSCALE IMAGE

# 3) Compute fourier transform
fft_gray = np.fft.fftshift(np.fft.fft2(img_gray))
fft_gray[:225, 75:85] = 1
fft_gray[-225:,75:85] = 1
display_image(np.log(abs(fft_gray)))
display_image(abs(np.fft.ifft2(fft_gray)))

ORDER = 3
highpass_gray = ski.filters.butterworth(
                img_gray,
                cutoff_frequency_ratio=0.1,
                order=ORDER,
                high_pass=True
            )

lowpass_gray = ski.filters.butterworth(
                img_gray,
                cutoff_frequency_ratio=0.1,
                order=ORDER,
                high_pass=False
            )
#display_image(highpass_gray)
