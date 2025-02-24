"""
Created on Fri Nov 29 11:03:20 2024
@author: guyadern
"""



import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import matplotlib.colors as colors

from skimage import data, io, color
from skimage.color import rgb2gray
from scipy.ndimage import convolve
from scipy.fftpack import fft2, fftshift

# %% An image is read and then displayed
# skimage.data contains several images that are used in skimage-related tutorials

im1 = data.camera()
# for a greyscale image : dim = 2
plt.figure()
io.imshow(im1)
axes = plt.gca()
axes.axis('off')
axes.set_title('Affichage par skimage')

plt.figure()
plt.imshow(im1)  # ou plt.imshow(im1,cmap=plt.cm.gray)
plt.colorbar()
axes = plt.gca()
axes.axis('off')
axes.set_title('Affichage par matplotlib')

#%%
im2 = data.cat()
# for a color image : dim = 3
plt.figure()
io.imshow(im2)
axes = plt.gca()
axes.axis('off')
axes.set_title('Affichage par skimage')


#%%
"""Note the differences between the two imshow functions (two libraries)
skimage displays the real data according to their value in ubytes 
and matplolib displays by normalising between the min and max of the image"""

# convert a greyscale image into a color image
im2_gl = rgb2gray(im2)
# Visualisation of each color channel
fig, axes = plt.subplots(1, 3, figsize=(8, 4))
axes[0].imshow(im2[:, :, 0])
axes[0].axis("off")
axes[1].imshow(im2[:, :, 1])
axes[1].axis("off")
axes[2].imshow(im2[:, :, 2])
axes[2].axis("off")
fig.tight_layout()
plt.show()



# greyscale view
fig1, axes1 = plt.subplots(1, 3, figsize=(8, 4))
axes1[0].imshow(im2[:, :, 0], cmap=plt.cm.gray)
axes1[0].axis("off")
axes1[1].imshow(im2[:, :, 1], cmap=plt.cm.gray)
axes1[1].axis("off")
axes1[2].imshow(im2[:, :, 2], cmap=plt.cm.gray)
axes1[2].axis("off")
fig1.tight_layout()
plt.show()


#%%
# Another visualisation
cmap = plt.cm.gray
R = np.copy(im2)
R[:, :, 1] = 0
R[:, :, 2] = 0

V = np.copy(im2)
V[:, :, 0] = 0
V[:, :, 2] = 0

B = np.copy(im2)
B[:, :, 0] = 0
B[:, :, 1] = 0

fig, ax = plt.subplots(3, 3, figsize=(12, 8))
fig.suptitle('Red, Green, Blue visualisation')

ax[0, 1].imshow(im2)
ax[0, 1].set_title('Color image')

ax[1, 0].imshow(R)
ax[1, 0].set_title('Red channel')

ax[1, 1].imshow(V)
ax[1, 1].set_title('Green channel')

ax[1, 2].imshow(B)
ax[1, 2].set_title('Blue channel')

ax[2, 0].imshow(im2[:, :, 0], cmap=cmap)
ax[2, 0].set_title('Red channel')

ax[2, 1].imshow(im2[:, :, 1], cmap=cmap)
ax[2, 1].set_title('Green channel')

ax[2, 2].imshow(im2[:, :, 2], cmap=cmap)
ax[2, 2].set_title('Blue channel')

for i in range(3):
    ax[0, i].axis('off')
    ax[1, i].axis('off')
    ax[2, i].axis('off')

# %% Fourier Transform

# with a greyscale image (note that for a color image the Fourier Transform is calculated for each channel)
im1 = data.camera()
fftI = fftshift(fft2(im1))

# Fourier Transform is a complex (see the "variable explorer")
# we can display the amplitude spectrum (module)
ASfftI = np.abs(fftI)
plt.figure()
plt.imshow(np.log(ASfftI))
plt.title('Amplitude spectrum ')

# have a look at https://scikit-image.org/docs/stable/auto_examples/filters/plot_window.html
# if we want to add a window before the Fourier Transform

# %% Image low-pass filtering
# see https://scikit-image.org/docs/stable/auto_examples/filters/plot_butterworth.html#


# %% Hybrid images
# see http://olivalab.mit.edu/publications/publications.html