"""preprocess.py
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
    """Computes center of mass. Scipy's center_of_mass function uses bright
    pixels to compute the center of mass, so we first invert the image.

    inputs:
    image -- The input image (PIL Image instance)

    outputs:
    cent_mass -- The center of mass of the input image
    """
    inverted_img = ImageOps.invert(image)
    inverted_arr = np.array(inverted_img)
    cent_mass = ndimage.measurements.center_of_mass(inverted_arr)
    return cent_mass

def crop_image(arr, cent_of_mass, crop_dims, args):
    """This function crops an image according to the center of mass and the
    crop dimensions. The center of mass will be the center of the image after
    cropping. The width and height will both be equal to crop_dims[0]*2 + 1.

    inputs:
    arr          -- The image to be cropped (numpy array)
    cent_of_mass -- The center of mass, will be the new image's center.
    crop_dims    -- The number of pixels on each side of the center pixel.
    args         -- The arguments to the program. Most of these are defaults.

    outputs:
    cropped      -- A cropped version of the input arr
    """
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
    """This function will zoom in on a character in an image.
    Essentially, this function calls crop_image again, with different 
    parameters, based on where the black pixels in the image are.
    All rows and columns outside the character are cropped away. Then, the
    character image is scaled so that the dimensions are the same.

    inputs:
    arr          -- The image numpy array
    cent_of_mass -- The center of mass. Should be equal to the center pixel
    crop_dims    -- The old number of pixels on each side of the center pixel.
                    The image after cropping will be scaled to have this number
                    of pixels on each side of the center pixel once more.
    args         -- The arguments to the program, mostly defaults.
    """
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
    """This function is called once when predicting sequences of character 
    labels in main.py. This is only used for predicting, because arrs_to_tensor
    is used when training.

    inputs:
    arrs         -- The sequence of character images, this sequence is a list
                    of lists of numpy arrays. Each of these numpy arrays
                    represents a character image. Each list of numpy arrays
                    represents a line of characters. Each list of lists
                    represents a whole document (multiple lines).
    args         -- The arguments to the program, mostly defaults.
    src_filename -- The filename of the image the characters come from.
                    This is only used for saving the visualized figures under
                    a comprehensive directory name.
    crop_dimensions -- An argument for the crop_image and zoom_image functions:
                       The numbers represent the number of pixels on each side
                       of the center pixel. If crop_dimensions == (1, 2), there
                       would be 1 pixel on the left and right of the center 
                       pixel, and 2 pixels on top and on bottom of the center
                       pixel.

    outputs:
    return_arrs  -- The sequence of character images, cropped according to
                    the center of mass, and zoomed. The sequence should have
                    the same structure as the input arrs.
    """
    if args.visualize:
        util.makedirs("Figures/cropped_chars")    

    return_arrs = [[] for _ in arrs]
    for l_idx, line in enumerate(arrs):
        for c_idx, char in enumerate(line):

            try:
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
            except Exception:
                print(f"An expection has occurred while segmenting {src_filename}. Skipping the problematic character!")
                continue

    return return_arrs

def arrs_to_tensor(arrs, args, crop_dimensions=(63, 63)):
    """This function preprocesses a numpy array of character images. This 
    function is only called in the train and test steps of main.py. For 
    predicting, we use preprocess_arrays. Essentially, the input images
    are cropped and zoomed, just like in preprocess_arrays, but then the
    sequence is converted to a Tensor object.

    inputs:
    arrs            -- The sequence of character images (numpy array)
    args            -- The arguments to the program, mostly defaults.
    crop_dimensions -- An argument for the crop_image and zoom_image functions:
                       The numbers represent the number of pixels on each side
                       of the center pixel. If crop_dimensions == (1, 2), there
                       would be 1 pixel on the left and right of the center 
                       pixel, and 2 pixels on top and on bottom of the center
                       pixel.

    outputs:
    torch.Tensor(return_arr) -- A PyTorch Tensor object of the preprocessed
                                character images.
    """
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