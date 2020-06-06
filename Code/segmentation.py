"""segmentation.py"""
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageOps as ImageOps
import PIL.ImageFilter as ImageFilter
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
import util
import time

def prepare_inverted_image(binarized_image, subsampling):
    # Open the image and invert it
    # We want to invert it for easier histogram-making     
    resized_image = binarized_image.resize((binarized_image.width//subsampling, binarized_image.height//subsampling))
    inverted_image = ImageOps.invert(resized_image)
    return inverted_image

def find_best_rotation(image, filename, args):
    """Find out what the minimal average number of black pixels (or white pixels in inverted images)
    is per row, for various rotated versions of the input test image
    MIGHTDO: Streamline this (Hough transform?)
    """
    min_avg = np.inf
    for rotation in range(-6, 6, 1):
    # for rotation in [1]: # found that 1 was the best in my test case
        rotated_image = image.rotate(rotation)
        minima_indices, avg_of_local_minima = line_segment(rotated_image, rotation, args.visualize, args.persistence_threshold, filename)
        if avg_of_local_minima < min_avg:
            min_avg = avg_of_local_minima
            best_rot = rotation
            best_minima_indices = minima_indices
    print(f"Best rotation: {best_rot}")
    return best_rot, best_minima_indices

def rotate_invert_image(image, rotation):
    rotated_image = image.rotate(rotation)
    inverted_rotated_image = ImageOps.invert(rotated_image)
    return inverted_rotated_image

def line_segment(image, rotation, visualize, persistence_threshold, filename):
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
    sorted_minima = extract_local_minima(histogram, persistence_threshold=persistence_threshold)
    if visualize:
        util.makedirs(f"../Figures/line_histograms/{filename}")
        vis.plot_histogram(histogram, f"../Figures/line_histograms/{filename}/smoothed_histogram_{rotation}")
        vis.plot_histogram(histogram, f"../Figures/line_histograms/{filename}/smoothed_histogram_with_extrema_{rotation}", minima=sorted_minima)


    # Some orientation might have different numbers of minima
    # To see how good the minima are, we average them.
    # We will work with the image orientation that has the lowest average of local minima
    # Since it is expected that the text lines are horizontal in that case.
    avg_of_local_minima = sum(histogram[sorted_minima])/len(sorted_minima)
    return sorted_minima, avg_of_local_minima


def create_histogram(image_array, smooth=15):
    """This function takes a binarized image,
    normalizes it and returns a 
    histogram of black pixels per row.
    """
    def normalize_mapping(x):
        return x//255

    def invert_mapping(x):
        return (x - 1) * -1

    arr = np.array(list(map(normalize_mapping, image_array)))

    if arr[0, 0] != 0:
        # Most likely, this array has not been inverted yet
        arr = np.array(list(map(invert_mapping, arr)))

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

def perform_astar_pathfinding(image_array, minima_rowindices, const_c, subsampling):

    def normalize_mapping(x):
        return x//255

    arr = np.array(list(map(normalize_mapping, image_array)))
    astar_paths = []

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    print(f"Computing A*-paths for rows: {minima_rowindices[1:-1]}")
    for index, row in enumerate(minima_rowindices[1:-1]):
        border_top = minima_rowindices[index]
        border_bot = minima_rowindices[index+2]
        p = multiprocessing.Process(target=astar, args=(arr, row, border_top, border_bot, return_dict, const_c, subsampling))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    astar_paths = return_dict.values()

    # Because we used multiple cores, the astar_paths are not sorted
    # So we sort them based on the y coordinate of the first element in each path
    for path in astar_paths.copy():
        _, y = path[0]
        correct_index = minima_rowindices.index(y // subsampling)
        astar_paths[correct_index - 1] = path

    return astar_paths

def astar(img_arr, line_num, border_top, border_bot, return_dict, const_c, subsampling):
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
                position = current.position * subsampling
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
                d_cost = const_c / (1 + min_dist_cost(img_arr, neighbour))
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

def extract_line_images(img_arr, astar_paths, n_cols, filename, args):
    segment_arrs = []
    for index, segment_bottom_path in enumerate(astar_paths[1:]):
        segment_top_path = astar_paths[index]
        # segment_top_path = supersample_path(segment_top_path)
        # segment_bottom_path = supersample_path(segment_bottom_path)

        top = img_arr.shape[1]
        bot = 0     
        # c, r -- because the path has x, y coordinate system instead of r, c
        for c, r in segment_top_path:
            if r < top:
                top = r
        for c, r in segment_bottom_path:
            if r > bot:
                bot = r

        n_rows = bot - top
        seg_arr = np.ones((n_rows, n_cols))
        seg_arr *= 255

        # copy the data from the image array between the segment lines
        # into the seg_arr
        for i in range(n_rows):
            r = i + top
            for c in range(n_cols):
                _ , top_row = segment_top_path[c]
                _ , bot_row = segment_bottom_path[c]
                if r >= top_row and r < bot_row:
                    seg_arr[r-top, c] = img_arr[r, c]

        segment_arrs.append(seg_arr)

        if args.visualize:
            segment_image = Image.fromarray(seg_arr).convert("L")
            util.makedirs(f"../Figures/line_segments/{filename}")
            save_location = f"../Figures/line_segments/{filename}/line_segment_{index}.png"
            segment_image.save(save_location, "PNG")
            print(f"Saved image to {save_location}")

    return segment_arrs

def segment_characters(line_segments, filename, args):
    astar_paths = []

    for index, seg_arr in enumerate(line_segments):
        # Steps to perform character zone segmentation:
        # Rotate 90 degrees
        seg_arr = np.rot90(seg_arr)

        # Perform histogram computation
        seg_hist = create_histogram(seg_arr, smooth=51)

        # Find local minima
        seg_minima = extract_local_minima(seg_hist, persistence_threshold=10)

        # Perform A*
        astar_path = perform_astar_pathfinding(seg_arr, seg_minima, args.CONST_C_CHAR, args.subsampling_char)
        astar_paths.append(astar_path)

        if args.visualize:
            util.makedirs([f"../Figures/char_histograms/{filename}", f"../Figures/astar_paths/{filename}"])
            vis.plot_histogram(seg_hist, f"../Figures/char_histograms/{filename}/character_histogram_{index}.png")
            image = Image.fromarray(seg_arr).convert("L")
            vis.draw_astar_lines(image, astar_path, width=3,
                                 save_location=f"../Figures/astar_paths/{filename}/char_segment_with_zones_{index}.png")

    return astar_paths

def extract_char_images(char_astar_paths, line_segments, filename, args):
    char_segments = []
    b=0
    for image in char_astar_paths: #go over every individual image
        for n in range(len(image)+1): #go over every line in the image plus one to get 4 areas for 3 lines that are saved
            max_x = 0
            min_y = 0
            line_array = []
            edit_line_array = []
            if n == (len(image)): # case when the area is form the top of the image to the previous red line
                min_y = len(line_segments[b][1]) # these values are used to initialize a 2d array with the sizes of the furthest outsticking line parts
                max_x = len(line_segments[b])
                try:
                    max_y = save_max_y
                except UnboundLocalError as err:
                    print(f"Skipping in extract_char_images due to error: {err}")
                    continue
            else:
                prev = np.inf
                # this for loop finds the dimensions of the array which copies the character
                for x in range(len(image[n])): 
                    if image[n][x][1] > min_y:
                        min_y = image[n][x][1]
                    if image[n][x][0] > max_x:
                        max_x = image[n][x][0]
                    if x > 0:
                        if image[n][x][0] != prev:
                            line_array.append(image[n][x])   #line_array saves all the line values that do not lie vertically above each other
                    else:
                       line_array.append(image[n][x])
                    prev = image[n][x][0]
                if n > 0: # if the area under any other than the first red line is to be saved
                    max_y = np.inf
                    for z in range(len(image[n-1])):
                        if image[n-1][z][1] < max_y:
                             max_y = image[n-1][z][1]
            if n > 0:
                save_max_y = max_y #save the highest laying value of this line
            if n == 0: #if it is the first red line
                array = [[255 for col in range(max_x+1)] for row in range(min_y)] #fill the array of the correct size with white pixel values
            elif n == len(image): #if it is the last red line
                array = [[255 for col in range(max_x+1)] for row in range(len(line_segments[b][0])-max_y)]
            else: #if it is any of the inbetween red lines
                array = [[255 for col in range(max_x+1)] for row in range(min_y-max_y)]

            for h in range(len(line_array)): #flip the y values since the image is turned on its side in line_segments
                dis = len(line_segments[b][0])-line_array[h][1]
                edit_line_array.append((line_array[h][0],dis))

            min_y = len(line_segments[b][0]) - min_y #flip this value as well because of the turning of the image


            n_black_pix = 0
            # go over every segment and save the pixel values in array

            for i in range(len(edit_line_array)): 
                if n == 0: #if it is the first red line
                    for j in range(edit_line_array[i][1],len(line_segments[b][0])): # goes over the y-values to and from the red lines
                        if line_segments[b][i][j] < 128:
                            array[j-min_y][i] = 0
                            n_black_pix += 1
                elif n == len(image): #if it is the last red line
                    for j in range(prev_line_array[i][1]):
                        if line_segments[b][i][j] < 128:
                            array[j][i] = 0
                            n_black_pix += 1
                else: #if it is any of the inbetween red lines
                    for j in range(edit_line_array[i][1],prev_line_array[i][1]):
                        if line_segments[b][i][j] < 128:
                            array[j-min_y][i] = 0
                            n_black_pix += 1

            prev_line_array = edit_line_array
            if n_black_pix < args.n_black_pix_threshold:
                # If we did not encounter n_black_pix_theshold black pixels, continue with the next segment
                continue

            array = np.asarray(array, dtype=np.uint8)

            # Rotate 90 degrees and flip the array for reconstruction of original orientation
            # MIGHTDO: Find out why the images are flipped and fix that, then remove the np.flipud here
            array = np.rot90(array)
            array = np.flipud(array)
            char_segments.append(array)

            char_image = Image.fromarray(array).convert("L") #convert the array to an image and save it

            if args.visualize:
                util.makedirs(f"../Figures/char_segments/{filename}")
                save_location = f"../Figures/char_segments/{filename}/char_segment_{b}_{n}.png"
                char_image.save(save_location, "PNG")
                print(f"Saved image to {save_location}")
        b += 1

    return char_segments
    
def segment_from_args(args, filename):
    if not args.visualize:
        print("Not visualizing intermediate results. Call this program with the option --visualize to visualize intermediate results.")
    else:
        fig_dirs = ["../Figures/char_segments", "../Figures/line_segments",
                    "../Figures/char_histograms", "../Figures/line_histograms"]
        util.makedirs(fig_dirs)

    binarized_image = Image.open(os.path.join(args.test_dataroot, filename))
    image = prepare_inverted_image(binarized_image, args.subsampling)
    best_rot, minima_indices = find_best_rotation(image, filename, args)
    image = rotate_invert_image(image, best_rot)

    if args.visualize:
        vis.draw_straight_lines(image, minima_indices)

    image_arr = np.array(image)
    astar_paths = perform_astar_pathfinding(image_arr, minima_indices, args.CONST_C, args.subsampling)
    astar_paths = supersample_paths(astar_paths)

    # We now have the A* paths in the horizontal direction
    if args.visualize:
        image = ImageOps.invert(binarized_image)
        image = image.rotate(best_rot)
        image = ImageOps.invert(image)
        vis.draw_astar_lines(image, astar_paths, save_location=f"../Figures/astar_paths/{filename}/astar_line_segments.png")


    image = prepare_inverted_image(binarized_image, 1)
    image = rotate_invert_image(image, best_rot)
    image_arr = np.array(image)
    image = Image.fromarray(image_arr)

    n_cols = len(astar_paths[0])

    # If we do not insert these dummy paths, we lose the first and last line segments
    dummy_top_path = [(i, 0) for i in range(n_cols)]
    dummy_bot_path = [(i, image.height - 1) for i in range(n_cols)]
    astar_paths.insert(0, dummy_top_path)
    astar_paths.append(dummy_bot_path)

    segmented_lines = extract_line_images(image_arr, astar_paths, n_cols, filename, args)
    char_astar_paths = segment_characters(segmented_lines, filename, args)
    segmented_characters = extract_char_images(char_astar_paths, segmented_lines, filename, args)
    return segmented_characters

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Segmentation, this file should not be directly run.")
    parser.add_argument("--test_dataroot", type=str, required=True, help="root-directory containing the testing set images")
    parser.add_argument("-v", "--visualize", action="store_true", help="Tell the program whether to visualize intermediate results. See visualizer.py")
    parser.add_argument("--CONST_C", type=int, default=-80, help="The constant C in the formula for D(n). See A* paper.")
    parser.add_argument("-s", "--subsampling", type=int, default=4, help="The subsampling factor of the test image prior to performing the A* algorithm.")
    parser.add_argument("--CONST_C_CHAR", type=int, default=-366, help="The constant C in the formula for D(n), used for segmenting characters. See A* paper.")
    parser.add_argument("-sc", "--subsampling_char", type=int, default=1, help="The subsampling factor of the segmented image prior to performing the A* algorithm (for characters).")
    parser.add_argument("-p", "--persistence_threshold", type=int, default=2, help="The persistence threshold for finding local extrema.")
    args = parser.parse_args()

    # A list of segmented character arrays will be returned
    all_segmented_characters = []

    test_filenames = os.listdir(args.test_dataroot)
    for filename in test_filenames:
        all_segmented_characters.append(segment_from_args(args, filename))
