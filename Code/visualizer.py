"""visualizer.py

The functions in this file are just for 
visualizing intermediate results only used
in the update presentations.
"""

import matplotlib.pyplot as plt
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import os
import numpy as np

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
    print(f"Saved histogram figure to {fig_filepath}.")
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
        os.mkdir(fig_dir)
        print(f"Made directory: {fig_dir}")


def draw_straight_lines(image, best_minima_rowindices, save_location="../Figures/best_line_segments.png"):
    """Draws straight lines on image that signify the local minima.
    Saves the drawn on image to the save_location, which is
    set to '../Figures/best_line_segments.png' by default.
    """
    drawer = ImageDraw.Draw(image)
    for line_y in best_minima_rowindices:
        drawer.line((0, line_y, image.width, line_y), fill=128, width=10)
    image.save(save_location, "PNG")
    print(f"Saved image to {save_location}")

def draw_astar_lines(image, astar_paths, save_location="../Figures/astar_line_segments.png"):
    """Draws paths on image that signify the segmented lines.
    Saves the drawn on image to the save_location, which is
    set to '../Figures/astar_line_segments.png' by default.
    """
    drawer = ImageDraw.Draw(image)
    for path in astar_paths:
        drawer.line(path, fill="#111111", width=10)
    image.save(save_location, "PNG")
    print(f"Saved image to {save_location}")
