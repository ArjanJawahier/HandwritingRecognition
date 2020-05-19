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
from scipy.signal import savgol_filter

import multiprocessing
from multiprocessing import Process, Queue

from persistence.persistence1d import RunPersistence
import visualizer as vis

import time # Debug

def prepare_inverted_image(binarized_image, subsampling_factor):
    # Open the image and invert it
    # We want to invert it for easier histogram-making     
    resized_image = binarized_image.resize((binarized_image.width//subsampling_factor, binarized_image.height//subsampling_factor))
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
    image_array = np.array(image)
    histogram = create_histogram(image_array)
    sorted_minima = extract_local_minima(histogram, persistence_threshold=10)
    if args.visualize:
        vis.plot_histogram(histogram, f"../Figures/smoothed_histogram_{rotation}")
        vis.plot_histogram(histogram, f"../Figures/smoothed_histogram_with_extrema_{rotation}", minima=sorted_minima)


    # Some orientation might have different numbers of minima
    # To see how good the minima are, we average them.
    # We will work with the image orientation that has the lowest average of local minima
    # Since it is expected that the text lines are horizontal in that case.
    avg_of_local_minima = sum(histogram[sorted_minima])/len(sorted_minima)
    return sorted_minima, avg_of_local_minima

def create_histogram(image_array, smooth=51):
    """This function takes a binarized image,
    normalizes it and returns a 
    histogram of black pixels per row.
    """
    def normalize_mapping(x):
        return x//255

    arr = np.array(list(map(normalize_mapping, image_array)))
    hist_list = []
    for row in arr:
        sum_black_pixels = np.sum(row)
        hist_list.append(sum_black_pixels)
    hist = np.array(hist_list)
    smooth_hist = savgol_filter(hist, smooth, 3)
    return smooth_hist

def extract_local_minima(histogram, persistence_threshold=10):
    """Extracts local minima from the histogram based on the persistence1d method.
    This was also done in the A* paper.
    """

    # Use persistence to find out local minima
    extrema_with_persistence = RunPersistence(histogram)

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

def perform_astar_pathfinding(args, image_array, minima_rowindices):

    def normalize_mapping(x):
        return x//255

    arr = np.array(list(map(normalize_mapping, image_array)))
    print(arr.shape)
    astar_paths = []

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for index, row in enumerate(minima_rowindices[1:-1]):
        border_top = minima_rowindices[index]
        border_bot = minima_rowindices[index+2]
        p = multiprocessing.Process(target=astar, args=(args, arr, row, border_top, border_bot, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    astar_paths = return_dict.values()

    # Because we used multiple cores, the astar_paths are not sorted
    # So we sort them based on the y coordinate of the first element in each path
    for path in astar_paths.copy():
        _, y = path[0]
        correct_index = minima_rowindices.index(y//args.subsampling)
        astar_paths[correct_index - 1] = path


    return astar_paths

def astar(args, img_arr, line_num, border_top, border_bot, return_dict):
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

        neighbours = get_neighbours(img_arr, current_node, line_num, border_top, border_bot)

        for neighbour_cost, neighbour in neighbours:
            inserted = False
            if neighbour not in expanded_nodes:
                # Calculate g, h and f values
                d_cost = args.CONST_C / (1 + min_dist_cost(img_arr, neighbour))
                neighbour.g = neighbour_cost + d_cost
                neighbour.h = 10 * np.linalg.norm(neighbour.position - end_node.position)
                neighbour.f = neighbour.g + neighbour.h

                if (neighbour not in top_options) or (top_options[neighbour.__hash__] > neighbour.f):
                    heappush(priority_queue , neighbour) 
                    top_options[neighbour.__hash__] = neighbour.f
                else:
                    print("Skip")
                

def get_neighbours(img_arr, current_node, line_num, border_top, border_bot):
    """Gets the neighbouring nodes together with the neighbour cost"""
    neighbours = []
    possible_moves = np.array([[0, 1], [1, 1], [-1, 1], [-1, 0], [1, 0]])
    for move in possible_moves:

        neighbour_pos = current_node.position + move
        r, c = neighbour_pos

        if r < 0 or r >= img_arr.shape[0] or c < 0 or c >= img_arr.shape[1]:
            # New position is out of bounds! Ignore this move
            continue

        if r <= border_top or r >= border_bot:
            # New position is out of bounds! Ignore this move
            continue

        new_node = Node(parent=current_node, position=neighbour_pos)
        if np.array_equal(move, np.array([1, 1])) or np.array_equal(move, np.array([-1, 1])):
            # Neighbour cost is 14 when moving diagonally
            # But 9 when moving diagonally towards the line_num
            if abs(r - line_num) > abs(current_node.position[0] - line_num):
                neighbours.append((14, new_node))
            else:
                # Testing this! TODO: EITHER REMOVE THIS OR LET IT STAY IF IT WORKS
                neighbours.append((9, new_node))
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

def supersample_paths(paths):
    """Calls the supersample_path function on a list of paths"""
    supersampled_paths = []
    for path in paths:
        path = supersample_path(path)
        supersampled_paths.append(path)
    return supersampled_paths

def supersample_path(path):
    """This function fills in the spaces between coordinate pairs in a path.
    If path = [(0,0), (0,4)], this function will return
    [(0,0), (0,1), (0,2), (0,3), (0,4)]"""
    coords_added = 0
    for index, coords in enumerate(path[1:]):
        # Check whether the next 
        prev_coords = path[index + coords_added]
        # r and c are swapepd in this path
        prev_c, prev_r = prev_coords
        c, r = coords
        # There are 5 cases (derived from the 5 possible moves an agent can make)
        # Case 1: The r coordinate is the same, but not the c
        if prev_r == r and c > prev_c:
            for i in range(1, c - prev_c):
                path.insert(index + coords_added + 1, (prev_c + i, prev_r))
                coords_added += 1

        # Case 2: The r coordinate is larger than the previous r
        elif prev_r > r and c > prev_c:
            for i in range(1, c - prev_c):
                path.insert(index + coords_added + 1, (prev_c + i, prev_r - i))
                coords_added += 1

        # Case 3: The r coordinate is smaller than the previous r
        elif prev_r < r and c > prev_c:
            for i in range(1, c - prev_c):
                path.insert(index + coords_added + 1, (prev_c + i, prev_r + i))
                coords_added += 1

        # Case 4 and 5 we dont need to supersample, since we only need 1 coordinate pair for each c
        # However, we still need to delete the previous coords in these cases
        elif prev_r != r and c == prev_c:
            path.pop(index + coords_added)
            coords_added -= 1

    return path

if __name__ == "__main__":
    # This is test code and should be removed later
    # After it works
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualize", action="store_true", help="Tell the program whether to visualize intermediate results. See visualizer.py")
    parser.add_argument("-t", "--testdata", default="../Test_Data", help="Location of test data (can be relative or absolute path)")
    parser.add_argument("--CONST_C", type=int, default=-80, help="The constant C in the formula for D(n). See A* paper.")
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
    inverted_image = prepare_inverted_image(binarized_image, args.subsampling)
    best_rot, minima_rowindices = find_best_rotation(inverted_image)
    image = rotate_invert_image(inverted_image, best_rot)

    # At this point, we have the best rotation for the input test image
    # And we also have the minima rowindices for rotated test image.
    print(minima_rowindices)

    # We can draw lines at the mimima rowindices in the rotated image
    if args.visualize:
        vis.draw_straight_lines(image, minima_rowindices)

    image_array = np.array(image)
    line_segments = perform_astar_pathfinding(args, image_array, minima_rowindices)
    line_segments = supersample_paths(line_segments)
    # We now have the A* paths in the horizontal direction,
    # which is our line segmentation
    if args.visualize:
        inverted_original_image = ImageOps.invert(binarized_image)
        rotated_original_image = inverted_original_image.rotate(best_rot)
        inverted_inverted_image = ImageOps.invert(rotated_original_image)
        vis.draw_astar_lines(inverted_inverted_image, line_segments)


    inverted_image = prepare_inverted_image(binarized_image, 1)
    image = rotate_invert_image(inverted_image, best_rot)
    image_array = np.array(image)
    image_from_array = Image.fromarray(image_array)
    # image_from_array.resize((image_from_array.width//4, image_from_array.height//4)).show()
    # If we do not insert these dummy paths, we lose the first and last line segments
    num_cols = len(line_segments[0])
    dummy_top_path = [(i, 0) for i in range(num_cols)]
    dummy_bot_path = [(i, image_array.shape[0] - 1) for i in range(num_cols)]
    line_segments.insert(0, dummy_top_path)
    line_segments.append(dummy_bot_path)


    segment_arrays = []
    for index, segment_bottom_path in enumerate(line_segments[1:]):
        segment_top_path = line_segments[index]
        # segment_top_path = supersample_path(segment_top_path)
        # segment_bottom_path = supersample_path(segment_bottom_path)

        top = image_array.shape[1]
        bot = 0     
        # c, r -- because the path has x, y coordinate system instead of r, c
        for c, r in segment_top_path:
            if r < top:
                top = r
        for c, r in segment_bottom_path:
            if r > bot:
                bot = r

        num_rows = bot - top
        segment_array = np.ones((num_rows, num_cols))
        segment_array *= 255

        # copy the data from the image array between the segment lines
        # into the segment_array
        for i in range(num_rows):
            r = i + top
            for c in range(num_cols):
                _ , top_row = segment_top_path[c]
                _ , bot_row = segment_bottom_path[c]
                if r >= top_row and r < bot_row:
                    segment_array[r-top, c] = image_array[r, c]

        segment_arrays.append(segment_array)

        if args.visualize:
            segment_image = Image.fromarray(segment_array).convert("RGB")
            # segment_image.resize((segment_image.width//4, segment_image.height//4)).show()
            save_location = f"../Figures/line_segment_{index}.png"
            segment_image.save(save_location, "PNG")
            print(f"Saved image to {save_location}")

    # Character segmentation starts here
    for index, segment_array in enumerate(segment_arrays):
        # Steps to perform character zone segmentation:
        # Rotate 90 degrees
        rotated_segment_array = np.rot90(segment_array)

        # Perform histogram computation
        segment_histogram = create_histogram(rotated_segment_array, smooth=51)

        # Find local minima
        segment_local_minima = extract_local_minima(segment_histogram, persistence_threshold=10)
        print(segment_local_minima)
        # Perform astar
        args.subsampling = 1
        zone_segments = perform_astar_pathfinding(args, rotated_segment_array, segment_local_minima)
        # Rotate -90 degrees

        if args.visualize:
            segment_array_image = Image.fromarray(rotated_segment_array).convert("RGB")
            vis.draw_astar_lines(segment_array_image, zone_segments, width=10,
                                 save_location=f"../Figures/line_segment_{index}_with_zones.png")