import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure


def image_pipe(image_file):
    image = cv2.imread(image_file, 0)
    _, hog_image = hog(image, orientations=18, pixels_per_cell=(4, 4),
                       cells_per_block=(1, 1), visualize=True, multichannel=False)
    image = exposure.rescale_intensity(hog_image, in_range=(0, 255))
    imageLINE = image.reshape(-1, 24 * 32).astype(np.float64)
    return imageLINE