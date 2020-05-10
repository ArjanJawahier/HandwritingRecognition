"""segmentation.py"""
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageOps as ImageOps
import matplotlib.pyplot as plt
import os
import numpy as np

from persistence.persistence1d import RunPersistence

def line_segment(binarized_image, rotation):
    """This function segments the binarized image
    into horizontal lines of text, using the A*
    algorithm outlined in:
    https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/LineSegmentation.pdf

    This function also uses the persistence1d module, downloaded from:
    https://www.csc.kth.se/~weinkauf/notes/persistence1d.html
    """

    # Create histogram
    histogram = create_histogram(binarized_image)
    plot_histogram(histogram, "../Figures/histogram_" + str(rotation))

    sorted_minima = extract_local_minima(histogram)
    plot_histogram(histogram, "../Figures/histogram_with_extrema_" + str(rotation), minima=sorted_minima)

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
    def mapping(x):
        return x//255

    arr = np.array(binarized_image)
    arr = np.array(list(map(mapping, arr)))
    hist_list = []
    for row in arr:
        sum_black_pixels = np.sum(row)
        hist_list.append(sum_black_pixels)
    hist = np.array(hist_list)
    return hist
    

def plot_histogram(hist, fig_filepath, minima=None):
    """Plots the given array of counts of black pixels as a
    horizontal bar chart. It then saves the image to the given
    fig_filepath.
    """
    create_figdir(fig_filepath)
    fig, ax = plt.subplots()
    ax.barh(np.arange(len(hist)), width=hist, height=5.0, color="black")
    if minima is not None:
        ax.barh(minima, width=np.max(hist), height=15.0, color="red")
    ax.set_xlabel("Num black pixels")
    ax.set_ylabel("Row")
    ax.set_title("Histogram of black pixels per row" if minima is None else "Histogram of black pixels per row + minima")
    ax.set_ylim(len(hist), 0)
    plt.savefig(fig_filepath)
    plt.close()

def create_figdir(fig_filepath):
    """This function creates a directory if it is not
    already present. The directory made is based on the
    fig_filepath given to plot_histogram.
    """
    split = fig_filepath.split("/")
    fig_dir = ""
    for x in split[:-1]:
        fig_dir += x + "/"

    if not os.path.isdir(fig_dir):
        print("Making" + fig_dir)
        os.mkdir(fig_dir)

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



if __name__ == "__main__":
    # This is test code and should be removed later
    # After it works
    test_dir = "../Test_Data"
    test_filenames = os.listdir(test_dir)
    for f in test_filenames:
        if "binarized" in f:
            filename = f
            break
    binarized_image = Image.open(os.path.join(test_dir, filename))
    inverted_image = ImageOps.invert(binarized_image)

    min_avg = np.inf
    # for rotation in range(-6, 6, 1):
    for rotation in [0]: # found that 5 was the best in this test case
        rotated_image = inverted_image.rotate(rotation)
        minima_rowindices, avg_of_local_minima = line_segment(rotated_image, rotation)
        if avg_of_local_minima < min_avg:
            min_avg = avg_of_local_minima
            best_rot = rotation
            best_minima_rowindices = minima_rowindices

    # At this point, we have the best rotation for the input test image
    # And we also have the minima rowindices for rotated test image.
    print(best_rot, best_minima_rowindices)

    # We can draw lines at the mimima rowindices in the rotated image
    rotated_image = inverted_image.rotate(best_rot)
    inverted_rotated_image = ImageOps.invert(rotated_image)
    draw = ImageDraw.Draw(inverted_rotated_image)
    for line_y in best_minima_rowindices:
        draw.line((0, line_y, inverted_rotated_image.width, line_y), fill=128, width=10)
    inverted_rotated_image.save("../Figures/no_rot_line_segment", "PNG")

