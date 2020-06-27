"""preprocess.py
The code in this file calculates the center of mass of an input image
and crops the image with the center of mass as the new center of the output image.
"""

from skimage.transform import resize
from scipy import ndimage
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import numpy as np
import os
import torch

import Code.util as util

def center_of_mass(image):
    """Computes center of mass.
    Scipy's center_of_mass function
    uses bright pixels to compute the center of mass,
    so we first invert the image"""
    inverted_img = ImageOps.invert(image)
    inverted_arr = np.array(inverted_img)
    cent_mass = ndimage.measurements.center_of_mass(inverted_arr)
    return cent_mass

def crop_image(arr, cent_of_mass, crop_dims, args):
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

def zoom_image(arr, cent_of_mass, crop_dims, args):
    # The width and height should be equal at this point
    assert(arr.shape[0] == arr.shape[1])
    # invert and normalize
    inverted = np.absolute(arr/255 - 1)
    top = np.min( np.where(inverted != 0)[0] )
    bot = np.max( np.where(inverted != 0)[0] )
    left = np.min( np.where(inverted != 0)[1] )
    right = np.max( np.where(inverted != 0)[1] )
    biggest_diff = max(right-left, bot-top)
    new_crop_dims = (biggest_diff//2, biggest_diff//2)

    # Crop original array
    center = (arr.shape[0]+1)/2, (arr.shape[1]+1)/2
    zoomed = crop_image(arr, center, new_crop_dims, args)
    img = Image.fromarray(zoomed).convert("L")
    scaled = img.resize(crop_dims, resample=1)
    scaled = np.array(scaled, dtype=np.uint8)
    binarized = np.where(scaled>128, 255, 0)
    binarized = binarized.astype(np.uint8)
    return binarized


def preprocess_arrays(arrs, args, src_filename=None, crop_dimensions=(63, 63)):
    if args.visualize:
        util.makedirs("Figures/cropped_chars")    

    return_arrs = [[] for _ in arrs]
    for l_idx, line in enumerate(arrs):
        for c_idx, char in enumerate(line):
            image = Image.fromarray(char).convert("L")
            cent_mass = center_of_mass(image)

            image_arr = np.array(image)
            image_arr = crop_image(image_arr, cent_mass, crop_dimensions, args)
            image_arr = zoom_image(image_arr, cent_mass, crop_dimensions, args)

            if np.sum(np.absolute(image_arr/255 - 1)) > args.n_black_pix_threshold:
                return_arrs[l_idx].append(image_arr)

            if args.visualize:
                image = Image.fromarray(image_arr).convert("L")
                util.makedirs(f"Figures/cropped_chars/{src_filename}")
                image.save(f"Figures/cropped_chars/{src_filename}/char_{l_idx}_{c_idx}.png")


    return return_arrs

def arrs_to_tensor(arrs, args, crop_dimensions=(63, 63)):
    # This expects numpy arrays
    return_arr = []
    for img in arrs:
        img = img.reshape(img.shape[-2], img.shape[-1]) * 255
        img = Image.fromarray(img).convert("L")
        cent_mass = center_of_mass(img)
        img = np.array(img, dtype=np.uint8)
        img = crop_image(img, cent_mass, crop_dimensions, args)
        img = zoom_image(img, cent_mass, crop_dimensions, args) / 255
        img = resize(img, (args.image_size, args.image_size))
        img = img.reshape(1, img.shape[0], img.shape[1]) 
        return_arr.append(img)
    return_arr = np.array(return_arr)
    return torch.Tensor(return_arr)