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
import time
import cv2 as cv

from Code.persistence.persistence1d import RunPersistence
import Code.visualizer as vis
import Code.util as util


def prepare_inverted_image(img, subsampling):
    """Resize and invert an input image. It is inverted for the purposes of 
    being able to sum the number of black pixels in the image.

    inputs:
    img -- The input image, which will be resized and inverted.

    outputs:
    inverted_image -- The resized and inverted image.
    """
    resized_image = img.resize((img.width//subsampling, img.height//subsampling))
    inverted_image = ImageOps.invert(resized_image)
    return inverted_image

def find_best_rotation(image, filename, args):
    """Find out what the minimal average number of black pixels (or white 
    pixels in inverted images) is per row, for various rotated versions of
    the input test image.

    inputs:
    image     -- The input image (test image).
    filename  -- The filename of the input test image, this is used for saving
                 figures that have to do with this image.
    args      -- The arguments to the program, mostly defaults.

    outputs:
    best_rot  -- The best rotation, based on the average number of black pixels
                 in the minima of the rotated image (degrees).
    best_minima_indices -- The row indices of the minima (spaces between text)
                           in the best rotated image.    
    """
    min_avg = np.inf
    for rotation in range(-6, 6, 1):
        rotated_image = image.rotate(rotation)
        minima_indices, avg_of_local_minima = extract_local_minima(
                                                  rotated_image,
                                                  rotation,
                                                  args.visualize,
                                                  args.persistence_threshold,
                                                  filename
                                              )
        if avg_of_local_minima < min_avg:
            min_avg = avg_of_local_minima
            best_rot = rotation
            best_minima_indices = minima_indices
    print(f"Best rotation: {best_rot}")
    return best_rot, best_minima_indices

def rotate_invert_image(image, rotation):
    """Rotates and inverts a given image, according to a given rotation.
    The rotation should be given in degrees.

    returns:
    inverted_rotated_image -- The inverted and rotated image.
    """
    rotated_image = image.rotate(rotation)
    inverted_rotated_image = ImageOps.invert(rotated_image)
    return inverted_rotated_image

def extract_local_minima(image, rot, visualize, persistence_threshold, filename):
    """This function segments the binarized image into horizontal lines of 
    text, using the A* algorithm outlined in:
    https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/LineSegmentation.pdf
    This function also uses the persistence1d module, downloaded from:
    https://www.csc.kth.se/~weinkauf/notes/persistence1d.html

    inputs:
    image     -- The image which we want to segment into lines.
    rot       -- The number of degrees the image was rotated. Useful for saving
                 visualized figures under a comprehensive name.
    visualize -- Whether to save figures based on this process.
    persistence_threshold -- The persistence threshold determines which minima
                             we are interested in. A very low threshold would
                             result in extracting all local minima, whereas a
                             very high threshold would result in only taking
                             very defined local minima.
    filename  -- The filename of the original input image, for saving figures.
    """

    # Create histogram
    image_array = np.array(image)
    histogram = create_histogram(image_array)
    sorted_minima = histogram_to_minima(
                        histogram,
                        persistence_threshold=persistence_threshold
                    )
    if visualize:
        util.makedirs(f"Figures/line_histograms/{filename}")
        vis.plot_histogram(
            histogram,
            f"Figures/line_histograms/{filename}/smoothed_histogram_{rot}"
        )
        vis.plot_histogram(
            histogram,
            f"Figures/line_histograms/{filename}/smoothed_histogram_with_extrema_{rot}",
            minima=sorted_minima
        )


    # Some orientation might have different numbers of minima
    # To see how good the minima are, we average them.
    # We work with the orientation that has the lowest average of local minima
    # Since it is expected that the text lines are horizontal in that case.
    avg_of_local_minima = sum(histogram[sorted_minima])/len(sorted_minima)
    return sorted_minima, avg_of_local_minima


def create_histogram(image_array, smooth=15):
    """This function takes a binarized image, normalizes it and returns a 
    histogram of black pixels for each row.

    inputs:
    image_array -- The numpy array representing the image.
    smooth      -- A factor determining the strength of the savgol filter.
                   The higher this number, the more the histogram is smoothed.

    outputs:
    smooth_hist -- A smoothed histogram, based on the number of black pixels
                   per row in the input image.
    """
    arr = image_array / 255
    if arr[0, 0] == 1:
        arr -= 1
        arr = np.absolute(arr)

    hist_list = []
    for row in arr:
        sum_black_pixels = np.sum(row)
        hist_list.append(sum_black_pixels)
    hist = np.array(hist_list)
    smooth_hist = savgol_filter(hist, smooth, 3)

    return smooth_hist

def histogram_to_minima(histogram, persistence_threshold=10):
    """Extracts local minima from the histogram based on the persistence1d 
    method. This was also done in the A* paper.

    inputs:
    histogram     -- The histogram from which we want to extract the minima.
                     For each row in the original input image, there is a value
                     of number of black pixels in this histogram.
    persistence_threshold -- The persistence threshold determines which minima
                             we are interested in. A very low threshold would
                             result in extracting all local minima, whereas a
                             very high threshold would result in only taking
                             very defined local minima.

    outputs:
    sorted_minima -- A sequence of the indices of the local minima.
    """

    # Use persistence to find out local minima
    extrema = RunPersistence(histogram)

    # Only take extrema with persistence > threshold
    filtered = [t[0] for t in extrema if t[1] > persistence_threshold]

    # Sort the extrema, results == [min, max, min, max, min, max, etc..]
    sorted_extrema = sorted(filtered)

    # Take every other entry, because we are only interested in local minima
    sorted_minima = sorted_extrema[::2]
    return sorted_minima


class Node:
    """A node class for A* Pathfinding."""

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
    """This function calls the astar function in a parallel manner, using
    the multiprocessing library. The A* algorithm is based on the details
    outlined in
    https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/LineSegmentation.pdf

    inputs:
    image_array -- The image array which we want to segment into lines.
    minima_rowindices -- The row indices of local minima (black pixels) in
                         the image. There will be exactly as many A* paths as
                         there are minima_rowindices.
    const_c -- A constant value which is used for the A* algorithm. See
               the paper for more details.

    outputs:
    astar_paths -- A sequence of A* paths, one for each local minimum.
    """
    arr = image_array / 255
    astar_paths = []

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    for index, row in enumerate(minima_rowindices[1:-1]):
        border_top = minima_rowindices[index]
        border_bot = minima_rowindices[index+2]
        p = multiprocessing.Process(
                target=astar,
                args=(arr, row, border_top, border_bot,
                      return_dict, const_c, subsampling)
            )
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
    """This function performs A* pathfinding to find a path between lines of 
    text. The code is largely based on 
    https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/LineSegmentation.pdf, but
    some changes were made.

    inputs:
    img_arr  -- The input image in which we perform the A* pathfinding.
    line_num -- The line where a local minimum was found. The start and end
                nodes of the A* algorithm have this number as vertical
                coordinate.
    border_top -- A number which says that the A* agent cannot cross this line.
    border_bot -- A number which says that the A* agent cannot cross this line.
    return_dict -- A sequence of A* paths. Each line_num gets an own entry.
                   This return_dict will also be returned, so each time this
                   function is called, the return_dict will grow in size.
    const_c -- A constant value which is used for the A* algorithm. See
               the paper for more details.

    outputs:
    return_dict -- A sequence of A* paths. Each line_num gets their own
                   A* path. A path should always be found for each line_num.
    """
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
        # If already checked (with higher priority) then take new one
        while (current_node in expanded_nodes):  
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
                

def get_neighbours(img_arr, current_node, line_num, border_top, border_bot):
    """Gets the neighbouring nodes (of a current node) together with the
    neighbour cost.

    inputs:
    img_arr -- The original img array.
    current_node -- The node which we want to find the best move for.
    line_num -- The vertical coordinate of the start and end nodes. This is
                relevant for our version of the algorithm. When the A* agent
                deviates from the line_num, the cost is higher than when
                the A* agent moves towards the line_num.
    border_top -- A line number which cannot be crossed by an agent.
    border_bot -- A line number which cannot be crossed by an agent.

    outputs:
    neighbours -- For each possible move, there will be a neighbouring node in
                  this sequence of neighbours. Here, each element is a tuple
                  of the form: (cost, new_node)
    """
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
            # But 9 when moving diagonally towards the line_num (experimental)
            if abs(r - line_num) > abs(current_node.position[0] - line_num):
                neighbours.append((14, new_node))
            else:
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
        if img_arr[r, current_c] < 1:
            breaked_up = True
            break
        else:
            dist_up += 1

    dist_down = 0
    for r in range(current_r, img_arr.shape[0]):
        if img_arr[r, current_c] < 1:
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


def extract_line_images(img_arr, astar_paths, n_cols, filename, visualize):
    """This function extracts line segments from the image. The segments
    are based on the A* paths.

    inputs:
    img_arr -- The numpy array of the test image.
    astar_paths -- The sequence of A* paths which detected paths between
                   textual lines.
    n_cols -- The number of columns the line images should have. This is based
              on the length of the A* paths inside the sequence astar_paths.
    filename -- The filename of the original img. Only used for saving figures.
    visualize -- Whether to save figures.

    outputs:
    segment_arrs -- A sequence of numpy arrays, where each numpy array
                    represents a segmented line.
    """
    segment_arrs = []
    for index, segment_bottom_path in enumerate(astar_paths[1:]):
        segment_top_path = astar_paths[index]

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

        if visualize:
            segment_image = Image.fromarray(seg_arr).convert("L")
            util.makedirs(f"Figures/line_segments/{filename}")
            save_location = f"Figures/line_segments/{filename}/line_segment_{index}.png"
            segment_image.save(save_location, "PNG")
            print(f"Saved image to {save_location}")

    return segment_arrs

def segment_characters(line_segments, filename, args):
    image_astar_paths = []

    for index, seg_arr in enumerate(line_segments):
        seg_arr = np.rot90(seg_arr)
        seg_hist = create_histogram(seg_arr, smooth=31)
        seg_minima = histogram_to_minima(seg_hist, persistence_threshold=args.persistence_threshold)

        line_astar_paths = perform_astar_pathfinding(
            seg_arr, seg_minima, 
            args.CONST_C_CHAR, args.subsampling_char
        )
        line_astar_paths = supersample_paths(line_astar_paths)

        # If we do not insert these dummy paths, we lose the first and last line segments
        dummy_top_path = [(i, 0) for i in range(seg_arr.shape[1])]
        dummy_bot_path = [(i, seg_arr.shape[0] - 1) for i in range(seg_arr.shape[1])]
        line_astar_paths.insert(0, dummy_top_path)
        line_astar_paths.append(dummy_bot_path)
        image_astar_paths.append(line_astar_paths)
        if args.visualize:
            util.makedirs([
                f"Figures/char_histograms/{filename}",
            ])
            vis.plot_histogram(
                seg_hist,
                f"Figures/char_histograms/{filename}/character_histogram_{index}.png"
            )

        if args.visualize_astar:
            util.makedirs([
                f"Figures/astar_paths/{filename}"
            ])
            image = Image.fromarray(seg_arr).convert("L")
            vis.draw_astar_lines(
                image, line_astar_paths, width=3,
                save_location=f"Figures/astar_paths/{filename}/char_segment_with_zones_{index}.png"
            )
    return image_astar_paths

def extract_char_images(char_astar_paths, line_segments, filename, args):
    """We can leverage the code from extracting lines. Extracting characters
    can be done with the same process. First, we rotate the line segment.
    Then, we extract the character segments like we did with the line segments.
    Then we rotate back. Meanwhile, we test if the character has enough ink.

    inputs:
    char_astar_paths -- Each line has numerous astar_paths that denote
                        boundaries between character zones
    line_segments -- The line segments with all the characters in it.
                     This is essentially a list of numpy arrays, where the
                     numpy arrays are the segmented line-images.
    filename -- The filename of the test image we are currently processing.
    args     -- The command line arguments (mostly defaults).

    outputs:
    char_segments -- A list of lists where each nested list contains all
                     segmented characters of a line.
    """
    l_idx = 0
    char_segments = []

    for segment, line_astar_paths in zip(line_segments, char_astar_paths):
        if len(line_astar_paths) == 0:
            continue

        n_cols = len(line_astar_paths[0])
        segment = np.rot90(segment)
        chars = extract_line_images(segment, line_astar_paths, n_cols, filename, visualize=False)

        # Remove characters without enough ink, according to the threshold
        chars_with_enough_ink = []
        for i, ch in enumerate(reversed(chars)):
            inverted_normalized = np.absolute(ch/255 - 1)
            if np.sum(inverted_normalized) > args.n_black_pix_threshold:
                ch = np.rot90(ch, 3)
                chars_with_enough_ink.append(ch)

        char_segments.append(chars_with_enough_ink)

        if args.visualize:
            for s_idx, char_segment in enumerate(chars_with_enough_ink):
                char_image = Image.fromarray(char_segment).convert("L")
                util.makedirs(f"Figures/char_segments/{filename}")
                save_location = f"Figures/char_segments/{filename}/char_segment_{l_idx}_{s_idx}.png"
                char_image.save(save_location, "PNG")
                print(f"Saved image to {save_location}")
        l_idx += 1
    return char_segments

def find_single_multi_char(img_arr, filename, args):
    output_arr = [[] for _ in img_arr]
    for idx, line in enumerate(img_arr):
        for l_idx, char_seg in enumerate(line):
            # Apply dilation and erosion to improve image
            arr = np.array(char_seg, dtype=np.uint8)
            kernel = np.ones((3,3),np.uint8)
            dilation = cv.dilate(arr,kernel,iterations = 3)
            erode = cv.erode(dilation,kernel,iterations = 2)
            inv_img = np.invert(erode)
            contours, h = cv.findContours(inv_img, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
            max_y_arr = []
            min_y_arr = []
            max_x_arr = []
            min_x_arr = []
            min_x = np.inf
            max_x = 0
            min_y = np.inf
            max_y = 0

            for k, cont in enumerate(contours):
                min_x = np.inf
                max_x = 0
                min_y = np.inf
                max_y = 0
                # Find the length of all the contours in the image
                for i, pixel in enumerate(cont): 
                    if pixel[0][0] < min_x:
                      min_x = pixel[0][0]
                    if pixel[0][0] > max_x:
                       max_x = pixel[0][0]
                    if pixel[0][1] < min_y:
                       min_y = pixel[0][1]
                    if pixel[0][1] > max_y:
                        max_y = pixel[0][1]
                max_x_arr.append(max_x)
                min_x_arr.append(min_x)
                max_y_arr.append(max_y)
                min_y_arr.append(min_y)

            # First check is the width of all contours in the original segment
            if len(max_x_arr) != 0 and len(min_x_arr) !=0 and max(max_x_arr) - min(min_x_arr) > 110: 
                if len(contours[0]) != 0:
                    for k, cont in enumerate(contours): #go over the different blobs in the original segment
                        if max_x_arr[k]-min_x_arr[k] != 0 and max_y_arr[k]-min_y_arr[k] != 0 and len(cont) > 110:
                            matrix = [[255 for x in range(max_x_arr[k]-min_x_arr[k]+2000)] for y in range(max_y_arr[k]-min_y_arr[k]+2000)] 
                            c = np.array(matrix, dtype=np.uint16)
                            c_UMat = cv.UMat(c)
                            cv.drawContours(c_UMat, contours, k, (0,0,0), -1)
                            c = cv.UMat.get(c_UMat)
                            c_inv = np.absolute(c/255 -1)
                            if np.sum(c_inv) > args.n_black_pix_threshold: 
                                output_arr[idx].append(c)
                                if args.visualize:
                                    if idx == 0:
                                        util.makedirs([
                                            f"Figures/split_char_images/{filename}"
                                        ])
                                    c_im = Image.fromarray(c).convert("L")
                                    save_loc = f"Figures/split_char_images/{filename}/split_char_image_{idx}_{l_idx}_{k}.png"
                                    c_im.save(save_loc, "PNG")
                                    print(f"Saved image to {save_loc}")
            else:
                output_arr[idx].append(arr)
                if args.visualize:
                    util.makedirs([
                        f"Figures/split_char_images/{filename}"
                    ])
                    c_im = Image.fromarray(erode).convert("L")
                    save_loc = f"Figures/split_char_images/{filename}/split_char_image_{idx}_{l_idx}.png"
                    c_im.save(save_loc, "PNG")
                    print(f"Saved image to {save_loc}")
    return output_arr

def segment_from_args(args, filename):
    """This is the function that calls all other function in this program.
    This function is called by main.py. 
    
    inputs:
    args -- The arguments to the program, mostly defaults.
    filename -- The filename of the image that we want to segment.
                The file still has to be opened, which is done in this function.
    
    outputs:
    finer_segmented_chars -- A list of lists of numpy arrays.
                             Each numpy array represents a character.
                             Each list of numpy arrays represents a line of
                             characters. Each list of lists of numpy arrays
                             represents all lines of characters in a document.
    """
    print(f"Segmenting {filename}")
    if args.visualize:
        fig_dirs = ["Figures/char_segments", "Figures/line_segments",
                    "Figures/char_histograms", "Figures/line_histograms"]
        util.makedirs(fig_dirs)

    binarized_image = Image.open(os.path.join(args.test_dataroot, filename))
    binarized_image = binarized_image.convert("L")
    image = prepare_inverted_image(binarized_image, args.subsampling)
    best_rot, minima_indices = find_best_rotation(image, filename, args)
    image = rotate_invert_image(image, best_rot)

    if args.visualize:
        vis.draw_straight_lines(image, minima_indices)

    image_arr = np.array(image)
    astar_paths = perform_astar_pathfinding(image_arr, minima_indices, args.CONST_C, args.subsampling)
    astar_paths = supersample_paths(astar_paths)

    # We now have the A* paths in the horizontal direction
    if args.visualize_astar:
        image = ImageOps.invert(binarized_image)
        image = image.rotate(best_rot)
        image = ImageOps.invert(image)
        vis.draw_astar_lines(image, astar_paths, save_location=f"Figures/astar_paths/{filename}/astar_line_segments.png")


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

    segmented_lines = extract_line_images(image_arr, astar_paths, n_cols, filename, args.visualize)
    char_astar_paths = segment_characters(segmented_lines, filename, args)
    segmented_characters = extract_char_images(char_astar_paths, segmented_lines, filename, args)
    finer_segmented_chars = find_single_multi_char(segmented_characters, filename, args)
    return finer_segmented_chars

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Segmentation, this file should not be directly run.")
    parser.add_argument("--test_dataroot", type=str, required=True, help="root-directory containing the testing set images")
    parser.add_argument("-v", "--visualize", action="store_true", help="Tell the program whether to visualize intermediate results. See visualizer.py")
    parser.add_argument("--CONST_C", type=int, default=-80, help="The constant C in the formula for D(n). See A* paper.")
    parser.add_argument("-s", "--subsampling", type=int, default=4, help="The subsampling factor of the test image prior to performing the A* algorithm.")
    parser.add_argument("--CONST_C_CHAR", type=int, default=-366, help="The constant C in the formula for D(n), used for segmenting characters. See A* paper.")
    parser.add_argument("-sc", "--subsampling_char", type=int, default=1, help="The subsampling factor of the segmented image prior to performing the A* algorithm (for characters).")
    parser.add_argument("-p", "--persistence_threshold", type=int, default=20, help="The persistence threshold for finding local extrema.")
    parser.add_argument("--n_black_pix_threshold", type=int, default=200, help="The minimum number of black pixels per character image.")

    args = parser.parse_args()

    # A list of segmented character arrays will be returned
    all_segmented_characters = []

    test_filenames = os.listdir(args.test_dataroot)
    for filename in test_filenames:
        all_segmented_characters.append(segment_from_args(args, filename))
