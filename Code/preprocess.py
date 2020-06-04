"""preprocess.py
The code in this file calculates the center of mass of an input image
and crops the image with the center of mass as the new center of the output image.
"""

from scipy import ndimage
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import numpy as np
import os
import util

def center_of_mass(image):
    """Computes center of mass.
    Scipy's center_of_mass function
    uses bright pixels to compute the center of mass,
    so we first invert the image"""
    inverted_img = ImageOps.invert(image)
    inverted_arr = np.array(inverted_img)
    cent_mass = ndimage.measurements.center_of_mass(inverted_arr)
    return cent_mass

def crop_image(arr, cent_of_mass, crop_dims):
    cent_r, cent_c = cent_of_mass
    cent_r, cent_c = int(cent_r), int(cent_c)
    
    cropped = np.ones((2*crop_dims[0]+1, 2*crop_dims[1]+1))
    cropped *= 255
    new_r = -1
    for r in range(cent_r-crop_dims[0], cent_r+crop_dims[1] + 1):
        new_r += 1

        if r < 0 or r >= arr.shape[0]:
            continue

        new_c = -1
        for c in range(cent_c-crop_dims[1], cent_c+crop_dims[1] + 1):
            new_c += 1

            if c < 0 or c >= arr.shape[1]:
                continue

            cropped[new_r, new_c] = arr[r, c]
    return cropped

def preprocess_arrays(arrs, src_filename, visualize=False, crop_dimensions=(63, 63)):
    if visualize:
        util.makedirs("../Figures/cropped_chars")    

    return_arrs = []
    for index, arr in enumerate(arrs):
        image = Image.fromarray(arr).convert("L")
        cent_mass = center_of_mass(image)

        image_arr = np.array(image)
        image_arr = crop_image(image_arr, cent_mass, crop_dimensions)
        return_arrs.append(image_arr)

        if visualize:
            image = Image.fromarray(image_arr).convert("L")
            util.makedirs(f"../Figures/cropped_chars/{src_filename}")
            # TODO: Change index to correct row and index in that row
            image.save(f"../Figures/cropped_chars/{src_filename}/char_{index}.png")
    return return_arrs


