"""edge_hinge.py
The code in this file calculates the edge hinge distribution from an image
"""

from skimage.transform import resize
from scipy import ndimage
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import numpy as np
import os
import util
import torch

def edge_hinge(img):
    frag_length = 4
    width, height = img.size
    eh = np.zeros(((frag_length - 1) * 8, (frag_length - 1) * 8))
    pic = np.array(img)
    for x in range(frag_length+1, width-frag_length-1):
        for y in range(frag_length+1, height-frag_length-1):
            count = 1
            found = False
            init = 1
            for z in range(0, frag_length-1):
                if not found:
                    if pic[y-z,x+frag_length] == 1:
                        found = True
                        init = count
                else:
                    if pic[y-z,x+frag_length] == 1:
                        eh[init, count] += 1
                count += 1
            for z in range(frag_length-2, -(frag_length - 1)+1, -1):
                if not found:
                    if pic[y-frag_length,x-z] == 1:
                        found = True
                        init = count
                else:
                    if pic[y-frag_length,x-z] == 1:
                        eh[init, count] += 1
                count += 1
            for z in range(frag_length-1, -(frag_length - 1)+1, -1):
                if not found:
                    if pic[y-z,x-frag_length] == 1:
                        found = True
                        init = count
                else:
                    if pic[y-z,x-frag_length] == 1:
                        eh[init, count] += 1
                count += 1
            for z in range(-(frag_length - 1), (frag_length - 2)-1):
                if not found:
                    if pic[y+frag_length,x+z] == 1:
                        found = True
                        init = count
                else:
                    if pic[y+frag_length,x+z] == 1:
                        eh[init, count] += 1
                count += 1
            for z in range(-(frag_length - 1), 0-1):
                if not found:
                    if pic[y-z,x+frag_length] == 1:
                        found = True
                        init = count
                else:
                    if pic[y-z,x+frag_length] == 1:
                        found = True
                        eh[init, count] += 1
                count += 1

    eh = eh / sum(sum(eh))

    return eh




if __name__ == "__main__":
    
    eh = []
    list_im = []
    for file in os.listdir("../Style_Data/Herodian/Alef/"):
        if file.endswith(".jpg"):
            img = Image.open(os.path.join("../Style_Data/Herodian/Alef/", file))
            eh.append(edge_hinge(img))
            list_im.append(os.path.join("../Style_Data/Herodian/Alef/", file))

    results    = [ Image.fromarray(i*100*255) for i in eh ]
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in results])[0][1]
    imgs_combh = np.hstack( (np.asarray( i.resize(min_shape) ) for i in results )) 

    imgs    = [ Image.open(i) for i in list_im ]
    imgs_combh2 = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    imgs_combv = np.vstack((imgs_combh, imgs_combh2))

    imgs_comb = Image.fromarray( imgs_combv).convert('RGB')
    imgs_comb.save("Herodian.png")

