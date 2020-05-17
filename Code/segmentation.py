"""segmentation.py"""
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageOps as ImageOps
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import argparse
from heapq import heapify, heappop, heappush

import multiprocessing
from multiprocessing import Process, Queue

from persistence.persistence1d import RunPersistence
import visualizer as vis

import time # Debug

def prepare_inverted_image(args, binarized_image):
    # Open the image and invert it
    # We want to invert it for easier histogram-making     
    resized_image = binarized_image.resize((binarized_image.width//args.subsampling, binarized_image.height//args.subsampling))
    inverted_image = ImageOps.invert(resized_image)
    return inverted_image

def find_best_rotation(image):
    """Find out what the minimal average number of black pixels (or white pixels in inverted images)
    is per row, for various rotated versions of the input test image
    TODO: Streamline this (Hough transform?)
    """
    min_avg = np.inf
    # for rotation in range(-6, 6, 1):
    for rotation in [5]: # found that 5 was the best in this test case
        rotated_image = inverted_image.rotate(rotation)
        minima_rowindices, avg_of_local_minima = line_segment(args, rotated_image, rotation)
        if avg_of_local_minima < min_avg:
            min_avg = avg_of_local_minima
            best_rot = rotation
            best_minima_rowindices = minima_rowindices
    return best_rot, minima_rowindices

def rotate_invert_image(image, rotation):
    rotated_image = image.rotate(rotation)
    inverted_rotated_image = ImageOps.invert(rotated_image)
    return inverted_rotated_image

def line_segment(args, image, rotation):
    """This function segments the binarized image
    into horizontal lines of text, using the A*
    algorithm outlined in:
    https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/LineSegmentation.pdf

    This function also uses the persistence1d module, downloaded from:
    https://www.csc.kth.se/~weinkauf/notes/persistence1d.html
    """

    # Create histogram
    histogram = create_histogram(image)
    sorted_minima = extract_local_minima(histogram)
    if args.visualize:
        vis.plot_histogram(histogram, f"../Figures/histogram_{rotation}")
        vis.plot_histogram(histogram, f"../Figures/histogram_with_extrema_{rotation}", minima=sorted_minima)


    # Some orientation might have different numbers of minima
    # To see how good the minima are, we average them.
    # We will work with the image orientation that has the lowest average of local minima
    # Since it is expected that the text lines are horizontal in that case.
    avg_of_local_minima = sum(histogram[sorted_minima])/len(sorted_minima)
    return sorted_minima, avg_of_local_minima

def create_histogram(image):
    """This function takes a binarized image,
    normalizes it and returns a 
    histogram of black pixels per row.
    """
    def normalize_mapping(x):
        return x//255

    arr = np.array(image)
    arr = np.array(list(map(normalize_mapping, arr)))
    hist_list = []
    for row in arr:
        sum_black_pixels = np.sum(row)
        hist_list.append(sum_black_pixels)
    hist = np.array(hist_list)
    return hist

def extract_local_minima(histogram):
    """Extracts local minima from the histogram based on the persistence1d method.
    This was also done in the A* paper.
    """

    # Use persistence to find out local minima
    extrema_with_persistence = RunPersistence(histogram)

    # Arbitrary persistence threshold (handcrafted)
    persistence_threshold = len(histogram) / 20

    # Only take extrema with persistence > threshold
    filtered_extrema = [t[0] for t in extrema_with_persistence if t[1] > persistence_threshold]

    # Sort the extrema, results == [min, max, min, max, min, max, etc..]
    sorted_extrema = sorted(filtered_extrema)

    # Take every other entry, because we are only interested in local minima
    sorted_minima = sorted_extrema[::2]
    return sorted_minima


class Node:
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __hash__(self):
        return hash(self.position.tobytes())

    def __eq__(self, other):
        return np.array_equal(self.position, other.position)

    def __lt__(self, other):
        return self.f < other.f

    def __repr__(self):
        return f"(pos: {self.position}, f: {self.f})"

def perform_astar_pathfinding(args, image, minima_rowindices):

    def normalize_mapping(x):
        return x//255

    arr = np.array(image)
    arr = np.array(list(map(normalize_mapping, arr)))
    print(arr.shape)
    astar_paths = []

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for row in range(len(minima_rowindices)):
        p = multiprocessing.Process(target=astar, args=(args, arr, minima_rowindices[row], return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    astar_paths = return_dict.values()

    return astar_paths

def astar(args, img_arr, line_num, return_dict):
    print(f"Computing A*-path for row: {line_num}")
    # The start node starts with H(n) = width of image
    # The start node start with F(n) = G'(n) + H(n)
    start_node = Node(parent=None, position=np.array([line_num, 0]))
    start_node.h = img_arr.shape[1]
    start_node.f = start_node.g + start_node.h
    end_node = Node(parent=None, position=np.array([line_num, img_arr.shape[1] - 1]))

    priority_queue = []
    heappush(priority_queue, start_node)
    expanded_nodes = set()

    while len(priority_queue) > 0:
        # First item in queue gets expanded first
        current_node = heappop(priority_queue)
        top_options = {}
        while (current_node in expanded_nodes):  # If already checked (with higher priority) then take new one
            current_node = heappop(priority_queue)
        expanded_nodes.add(current_node)


        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                position = current.position * args.subsampling
                r, c = position
                position = (c, r)
                path.append(position)
                current = current.parent
            return_dict[line_num] = path[::-1]
            break

        neighbours = get_neighbours(img_arr, current_node)

        for neighbour_cost, neighbour in neighbours:
            inserted = False
            if neighbour not in expanded_nodes:
                # Calculate g, h and f values
                d_cost = args.CONST_C / (1 + min_dist_cost(img_arr, neighbour))
                neighbour.g = neighbour_cost + d_cost
                neighbour.h = np.linalg.norm(end_node.position - neighbour.position)
                neighbour.f = neighbour.g + neighbour.h

                if (neighbour not in top_options) or (top_options[neighbour.__hash__] > neighbour.f):
                    heappush(priority_queue , neighbour) 
                    top_options[neighbour.__hash__] = neighbour.f
                else:
                    print("Skip")
                

def get_neighbours(img_arr, current_node):
    """Gets the neighbouring nodes together with the neighbour cost"""
    neighbours = []
    possible_moves = np.array([[0, 1], [-1, 0], [1, 0], [1, 1], [-1, 1]])
    for move in possible_moves:

        neighbour_pos = current_node.position + move
        r, c = neighbour_pos

        if r < 0 or r >= img_arr.shape[0] or c < 0 or c >= img_arr.shape[1]:
            # New position is out of bounds! Ignore this move
            continue

        new_node = Node(parent=current_node, position=neighbour_pos)
        if np.array_equal(move, np.array([1, 1])) or np.array_equal(move, np.array([-1, 1])):
            # Neighbour cost is 14 when moving diagonally
            neighbours.append((14, new_node))
        else:
            # Neighbour cost is 10 when moving up, down or to the right
            neighbours.append((10, new_node))
    return neighbours


def min_dist_cost(img_arr, node):
    """Calculate the minimum distance cost as defined in the paper:
    https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/LineSegmentation.pdf
    """
    max_value = 16384
    breaked_up = breaked_down = False
    dist_up = 0
    current_r, current_c = node.position
    for r in range(current_r, -1, -1):
        if img_arr[r, current_c] > 0:
            breaked_up = True
            break
        else:
            dist_up += 1

    dist_down = 0
    for r in range(current_r, img_arr.shape[0]):
        if img_arr[r, current_c] > 0:
            breaked_down = True
            break
        else:
            dist_down += 1


    if not breaked_up and not breaked_down:
        return max_value
    else:
        return min(dist_up, dist_down)

if __name__ == "__main__":
    # This is test code and should be removed later
    # After it works
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualize", action="store_true", help="Tell the program whether to visualize intermediate results. See visualizer.py")
    parser.add_argument("-t", "--testdata", default="../Test_Data", help="Location of test data (can be relative or absolute path)")
    parser.add_argument("--CONST_C", type=int, default=-50, help="The constant C in the formula for D(n). See A* paper.")
    parser.add_argument("-s", "--subsampling", type=int, default=4, help="The subsampling factor of the test image prior to performing the A* algorithm.")
    args = parser.parse_args()
    if not args.visualize:
        print("Not visualizing intermediate results. Call this program with the option --visualize to visualize intermediate results.")

    # Get an example test image (arbitrarily picked)
    test_filenames = os.listdir(args.testdata)
    for f in test_filenames:
        if "binarized" in f:
            filename = f
            break

    binarized_image = Image.open(os.path.join(args.testdata, filename))
    inverted_image = prepare_inverted_image(args, binarized_image)
    best_rot, minima_rowindices = find_best_rotation(inverted_image)
    image = rotate_invert_image(inverted_image, best_rot)

    # At this point, we have the best rotation for the input test image
    # And we also have the minima rowindices for rotated test image.
    print(minima_rowindices)

    # We can draw lines at the mimima rowindices in the rotated image
    if args.visualize:
        vis.draw_straight_lines(image, minima_rowindices)

    astar_paths = perform_astar_pathfinding(args, image, minima_rowindices)

    # We now have the A* paths, which is our line segmentation
    if args.visualize:
        inverted_original_image = ImageOps.invert(binarized_image)
        rotated_original_image = inverted_original_image.rotate(best_rot)
        inverted_inverted_image = ImageOps.invert(rotated_original_image)
        vis.draw_astar_lines(inverted_inverted_image, astar_paths)

    # We can now segment each line into character zones