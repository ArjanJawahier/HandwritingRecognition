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
import _pickle as pickle


class StyleClassifier:

    def __init__(self, train_data="../Style_Data/", use_dict="../style_data.pkl"):
        
        self.train_data = train_data
        if len(use_dict) == 0:
            print("Converting style data to average styles")
            self.average_styles = self.make_average_styles()

            dict_file = open("style_data.pkl", "wb")
            pickle.dump(self.average_styles, dict_file)
            dict_file.close()
            print("Created new dict file in Code/style_data.pkl")

        else:
            dict_file = open(use_dict, "rb")
            self.average_styles = pickle.load(dict_file)
            dict_file.close()

    def make_average_styles(self):
        """This function is used to create a dict of the edge hinges per 
        character per style.

        inputs:
        self -- The StyleClassifier

        outputs:
        avrage_styles -- A dict of an average style per character per style
        """
        average_styles = {}
        for style in os.listdir(self.train_data):
            for character in os.listdir(self.train_data + style + "/"):
                eh = []
                if len(os.listdir(self.train_data + style + "/" + character + "/")) < 3:
                    print(style, character)
                else:
                    for file in os.listdir(self.train_data + style + "/" + character + "/"):
                        if file.endswith(".jpg"):
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

    def predict_style(self, eh, character):
        """This function uses the distances between the trained style averages,
        and the edge hinge of the current character, to predict a style. 

        inputs:
        self -- The StyleClassifier
        eh -- The edge hinge matrix of the current character
        character -- The current character

        outputs:
        closest_style -- The predicted style, with the shortest distance
        """
        min_dist = np.inf
        for key, value in self.average_styles[character].items():
            dist = 0.5 * np.sum([((a - b) ** 2) / (a + b + 1e-6) for (a, b) in zip(eh, value)]) 
            if dist < min_dist:
                closest_style = key
                min_dist = dist
        # If None there is no average style for this character
        assert closest_style != None
        return closest_style

    def get_distance(self, eh, character):
        """This function gets the distances between the trained style averages,
        and the edge hinge of the current character. 

        inputs:
        self -- The StyleClassifier
        eh -- The edge hinge matrix of the current character
        character -- The current character

        outputs:
        distances -- An array of distances for each style
        """
        distances = {}
        for key, value in self.average_styles[character].items():
            distances[key] = 0.5 * np.sum([((a - b) ** 2) / (a + b + 1e-6) for (a, b) in zip(eh, value)])
            # distances[key] = np.linalg.norm(eh - value)
        return distances

    def edge_hinge(self, img):
        """This function calculates the edge hinge.
        It does this by looking at each point in the square surrounding a point
        in the array, If at one of these points 

        inputs:
        self -- The StyleClassifier
        img --The currect character image

        outputs:
        eh -- a matrix of values representing the edge hinge
        """
        frag_length = 5
        width, height = img.size
        eh = np.zeros(((frag_length) * 8 + 1, (frag_length) * 8 + 1))
        pic = np.array(img)
        for x in range(frag_length, width - frag_length):
            for y in range(frag_length, height - frag_length):
                count = 1
                found = False
                init = 1
                for z_1 in range(-frag_length, frag_length + 1):
                    for z_2 in range(-frag_length, frag_length + 1):
                        if (z_1 == -frag_length or z_1 == frag_length or z_2 == -frag_length or z_2 == frag_length):
                            if not found:
                                if pic[y + z_2, x + z_1] == 0:
                                    found = True
                                    init = count
                            else:
                                if pic[y + z_2, x + z_1] == 0:
                                    found = True
                                    eh[init, count] += 1
                            count += 1
        if sum(sum(eh)) > 0:
            eh = eh / sum(sum(eh))

        return eh

if __name__ == "__main__":
    """The main of this file is only present for testing purposes,
    The below code checks for each first character in each style of the
    Style Data which style it fits to.
    """
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
                    cs = classifier.predict_style(img, character)
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