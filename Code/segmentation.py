"""segmentation.py"""
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageOps as ImageOps
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import argparse

from persistence.persistence1d import RunPersistence
import visualizer as vis

def line_segment(binarized_image, rotation, args):
    """This function segments the binarized image
    into horizontal lines of text, using the A*
    algorithm outlined in:
    https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/LineSegmentation.pdf

    This function also uses the persistence1d module, downloaded from:
    https://www.csc.kth.se/~weinkauf/notes/persistence1d.html
    """

    # Create histogram
    histogram = create_histogram(binarized_image)
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

def create_histogram(binarized_image):
    """This function takes a binarized image,
    normalizes it and returns a 
    histogram of black pixels per row.
    """
    def normalize_mapping(x):
        return x//255

    arr = np.array(binarized_image)
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

    # TODO: Might need to cut off the white space above and under the actual text
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


class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def astar(img_arr, line_num, width):
    """Returns a list of tuples as a path from the given start to the given end in the given image"""

    # Create start and end node
    start_node = Node(None, (0, line_num))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, ((len(img_arr[0])-1), line_num))
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # print(current_node.position)
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        children = get_neighbours(img_arr, current_node)

        # Loop through children
        for child in children:
            valid = True

            # Child is on the closed list
            for closed_child in closed_list:
                if child.position == closed_child.position:
                    valid = False
                    continue


            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child.position == open_node.position and child.g >= open_node.g:
                    valid = False
                    continue

            # Add the child to the open list
            if valid:
                open_list.append(child)


def get_neighbours(img_arr, current_node):
    children = []
    for new_position in [(0, -1), (0, 1), (1, 0), (1, -1), (1, 1)]: # Adjacent squares

        # Get node position
        node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

        # Make sure within range
        if node_position[1] > (len(img_arr) - 1) or node_position[1] < 0 or node_position[0] > (len(img_arr[len(img_arr)-1]) -1) or node_position[0] < 0:
            continue

        # Make sure the new_position is walkable terrain
        if invalid_pixel(img_arr, current_node.position):
            continue

        # Create new node
        new_node = Node(current_node, node_position)

        # Append
        children.append(new_node)
    return children


def invalid_pixel(img_arr, current_pos):
    # TODO question: Why are we checking whether the current pos is an invalid
    # position? Shouldn't we check that before making it the current pos?
    # TODO question: Why are we adding x from a range of -5 to 4? Why are we checking backwards?
    for add_x in range(-5, 5):
        for add_y in range(-5, 5):
            # Get node position
            node_position = ((current_pos[0] + add_x), (current_pos[1] + add_y))

            # Make sure within range
            if node_position[1] > (len(img_arr) - 1) or node_position[1] < 0 or node_position[0] > (len(img_arr[len(img_arr)-1]) -1) or node_position[0] < 0:
                continue

            # Make sure walkable terrain
            if img_arr[node_position[1]][node_position[0]] != 0:
                return True

    return False


if __name__ == "__main__":
    # This is test code and should be removed later
    # After it works
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualize", action="store_true", help="Tell the program whether to visualize intermediate results. See visualizer.py")
    parser.add_argument("-t", "--testdata", default="../Test_Data", help="Location of test data (can be relative or absolute path)")
    args = parser.parse_args()
    if not args.visualize:
        print("Not visualizing intermediate results. Call this program with the option --visualize to visualize intermediate results.")

    # Get an example test image (arbitrarily picked)
    test_filenames = os.listdir(args.testdata)
    for f in test_filenames:
        if "binarized" in f:
            filename = f
            break

    # Open the image and invert it
    # We want to invert it for easier histogram-making     
    binarized_image = Image.open(os.path.join(args.testdata, filename))
    inverted_image = ImageOps.invert(binarized_image)

    # Find out what the minimal average number of black pixels (or white pixels in inverted images)
    # is per row, for various rotated versions of the input test image
    # TODO: Streamline this (Hough transform?)
    min_avg = np.inf
    # for rotation in range(-6, 6, 1):
    for rotation in [5]: # found that 5 was the best in this test case
        rotated_image = inverted_image.rotate(rotation)
        minima_rowindices, avg_of_local_minima = line_segment(rotated_image, rotation, args)
        if avg_of_local_minima < min_avg:
            min_avg = avg_of_local_minima
            best_rot = rotation
            best_minima_rowindices = minima_rowindices

    # At this point, we have the best rotation for the input test image
    # And we also have the minima rowindices for rotated test image.
    # print(best_rot, best_minima_rowindices)

    # We can draw lines at the mimima rowindices in the rotated image
    rotated_image = inverted_image.rotate(best_rot)
    inverted_rotated_image = ImageOps.invert(rotated_image)
    if args.visualize:
        vis.draw_straight_lines(inverted_rotated_image, best_minima_rowindices)

    def normalize_mapping(x):
        return x//255

    def swap_mapping(item):
        x, y = item
        return (y, x)

    arr = np.array(rotated_image)
    arr = np.array(list(map(normalize_mapping, arr)))
    astar_paths = []
    for row in best_minima_rowindices:
        print(f"Computing A*-path for row: {row}")
        astar_res = astar(arr, row, rotated_image.width)

        # What is this used for @Xabi?
        astar_result = np.array(list(map(swap_mapping, astar_res)))

        astar_paths.append(astar_res)

    # We now have the A* paths, which is our line segmentation

    if args.visualize:
        vis.draw_astar_lines(inverted_rotated_image, astar_paths)