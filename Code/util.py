"""util.py
"""
import os

def makedirs(dirs):
    if not isinstance(dirs, list):
        dirs = [dirs]

    for directory in dirs:
        if not os.path.exists(directory):
            print(f"Making directory {directory}")
            os.makedirs(directory)
