"""edge_hinge.py
The code in this file calculates the edge hinge distribution from an image
"""

from skimage.transform import resize
from scipy import ndimage
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import numpy as np
import os
import torch
import math

import Code.util as util


class StyleClassifier:

    def __init__(self, train_data="../Style_Data/"):
        self.train_data = train_data
        self.average_styles = self.make_average_styles()

    def make_average_styles(self):
        average_styles = {}
        for style in os.listdir(self.train_data):
            for character in os.listdir(self.train_data + style + "/"):
                eh = []
                if len(os.listdir(self.train_data + style + "/" + character + "/")) < 3:
                    print(style, character)
                else:
                    for file in os.listdir(self.train_data + style + "/" + character + "/"):
                        if file.endswith(".jpg") and not ("_00" in file):
                            img = Image.open(os.path.join(
                                self.train_data + style + "/" + character + "/", file))
                            eh.append(self.edge_hinge(img))
                    average = np.mean(eh, axis=0)
                    try:
                        average_styles[character][style] = average
                    except:
                        average_styles[character] = {}
                        average_styles[character][style] = average
        return average_styles

    def predict_style(self, img):
        eh = self.edge_hinge(img)
        min_dist = 99999
        for key, value in self.average_styles[character].items():
            dist = np.linalg.norm(eh - value)
            if dist < min_dist:
                closest_style = key
                min_dist = dist
        # If style is "" then there is no average style for this character
        assert closest_style != None
        return closest_style

    def edge_hinge(self, img):
        frag_length = 5
        width, height = img.size
        eh = np.zeros(((frag_length - 1) * 8, (frag_length - 1) * 8))
        pic = np.array(img)
        for x in range(frag_length, width - frag_length):
            for y in range(frag_length, height - frag_length):
                count = 1
                found = False
                init = 1
                for z_1 in range(-frag_length, frag_length):
                    for z_2 in range(-frag_length, frag_length):
                        if (z_1 == -frag_length or z_1 == frag_length or z_2 == -frag_length or z_2 == frag_length):
                            if not found:
                                if pic[y + z_2, x + z_1] == 1:
                                    found = True
                                    init = count
                            else:
                                if pic[y + z_2, x + z_1] == 1:
                                    found = True
                                    eh[init, count] += 1
                            count += 1

        if sum(sum(eh)) > 0:
            eh = eh / sum(sum(eh))

        return eh


if __name__ == "__main__":
    classifier = StyleClassifier("../Style_Data/")
    correct = 0
    incorrect = 0
    result_dict = {}
    character_dict = {}
    for style in os.listdir("../Style_Data/"):
        for character in os.listdir("../Style_Data/" + style + "/"):
            for file in os.listdir("../Style_Data/" + style + "/" + character + "/"):
                if file.endswith(".jpg") and ("_00" in file):
                    img = Image.open(os.path.join(
                        "../Style_Data/" + style + "/" + character + "/", file))
                    cs = classifier.predict_style(img)
                    try:
                        correct_char, incorrect_char = character_dict[character]
                    except:
                        correct_char, incorrect_char = (0, 0)
                    if cs == style:
                        character_dict[character] = (
                            correct_char + 1, incorrect_char)
                        correct += 1
                    else:
                        character_dict[character] = (
                            correct_char, incorrect_char + 1)
                        incorrect += 1
                    try:
                        result_dict[style][cs] += 1
                    except:
                        try:
                            result_dict[style][cs] = 1
                        except:
                            result_dict[style] = {}
                            result_dict[style][cs] = 1

    print(correct)
    print(incorrect)

    print(result_dict)
    print(character_dict)
