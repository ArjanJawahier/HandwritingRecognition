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

def line_segment(image, rotation, args):
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

    def __eq__(self, other):
        return np.array_equal(self.position, other.position)

def astar(img_arr, line_num, args):

    # TODO IDEA: Maybe we can maxpool the img_arr first, 
    # TODO IDEA: find the A*-path in the maxpooled image
    # TODO IDEA: and then we can scale the A* path up?

    # The start node starts with H(n) = width of image
    # The start node start with F(n) = G'(n) + H(n)
    start_node = Node(parent=None, position=np.array([0, line_num]))
    start_node.h = img_arr.shape[1]
    start_node.f = start_node.g + start_node.h

    end_node = Node(parent=None, position=np.array([img_arr.shape[1] - 1, line_num]))

    priority_queue = [start_node]
    expanded_nodes = []

    while len(priority_queue) > 0:
        # First item in queue gets expanded first
        current_node = priority_queue.pop(0)
        expanded_nodes.append(current_node)

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(tuple(current.position))
                current = current.parent
            return path[::-1]

        neighbours = get_neighbours(img_arr, current_node)

        for neighbour_cost, neighbour in neighbours:
            valid = True

            for node in expanded_nodes:
                if neighbour == node:
                    valid = False
                    break

            if valid:
                # Calculate g, h and f values
                d_cost = args.CONST_C / (1 + min_dist_cost(img_arr, current_node))
                neighbour.g = neighbour_cost + d_cost
                neighbour.h = np.linalg.norm(neighbour.position - end_node.position)
                neighbour.f = neighbour.g + neighbour.h

                for node in priority_queue:
                    if node.f > neighbour.f:
                        # If the cost is higher than the neighbour, we can stop the loop
                        # Since it's a priority queue, the list will be ordered on the nodes' f cost
                        break

                    if np.array_equal(neighbour.position, node.position) and neighbour.f < node.f:
                        priority_queue.remove(node)

                    for index, node in enumerate(priority_queue):
                        if node.f > neighbour.f:
                            # Insert the node at the right place in the queue
                            priority_queue.insert(index, neighbour)
                            break
                    break

                # Could be that the previous for loop didnt loop cause the queue was empty
                # Fix that by inserting the neighbour from within an if-statement
                if len(priority_queue) == 0:
                    priority_queue.insert(0, neighbour)

    if len(priority_queue) == 0:
        print("Test")
        path = []
        current = current_node
        while current is not None:
            path.append(tuple(current.position))
            current = current.parent
        return path[::-1]


def get_neighbours(img_arr, current_node):
    """Gets the neighbouring nodes together with the neighbour cost"""
    neighbours = []
    possible_moves = np.array([[0, 1], [-1, 1], [1, 1], [-1, 0], [1, 0]])
    for move in possible_moves:

        neighbour_pos = current_node.position + move
        r, c = neighbour_pos

        if r < 0 or r >= img_arr.shape[0] or c < 0 or c >= img_arr.shape[1]:
            # New position is out of bounds! Ignore this move
            continue

        new_node = Node(parent=current_node, position=neighbour_pos)
        if np.sum(move) != 1:
            # Neighbour cost is 14
            neighbours.append((14, new_node))
        else:
            neighbours.append((10, new_node))
    return neighbours


def min_dist_cost(img_arr, current_node):
    """Calculate the minimum distance cost as defined in the paper:
    https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/LineSegmentation.pdf
    """
    max_value = 16384
    dist_up = -1
    current_r, current_c = current_node.position
    for r in range(current_r, -1, -1):
        if img_arr[r, current_c] > 0:
            break
        else:
            dist_up += 1

    dist_down = -1
    for r in range(current_r, img_arr.shape[0]):
        if img_arr[r, current_c] > 0:
            break
        else:
            dist_down += 1

    if dist_up == -1 and dist_down == -1:
        return max_value
    else:
        return min(dist_up, dist_down)


if __name__ == "__main__":
    # This is test code and should be removed later
    # After it works
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualize", action="store_true", help="Tell the program whether to visualize intermediate results. See visualizer.py")
    parser.add_argument("-t", "--testdata", default="../Test_Data", help="Location of test data (can be relative or absolute path)")
    parser.add_argument("--CONST_C", type=int, default=250, help="The constant C in the formula for D(n). See A* paper.")
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
    resized_image = binarized_image.resize((binarized_image.width//4, binarized_image.height//4))
    inverted_image = ImageOps.invert(resized_image)

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

    arr = np.array(rotated_image)
    arr = np.array(list(map(normalize_mapping, arr)))
    print(arr.shape)
    astar_paths = []
    for row in best_minima_rowindices:
        print(f"Computing A*-path for row: {row}")
        astar_res = astar(arr, row, args)
        astar_paths.append(astar_res)

    # We now have the A* paths, which is our line segmentation
    if args.visualize:
        vis.draw_astar_lines(inverted_rotated_image, astar_paths)