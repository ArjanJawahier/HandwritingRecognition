# Handwriting Recognition
Handwriting Recognition project

## Dependencies
We are using quite a number of dependencies, use the following command to install them:  
```bash
pip3 install numpy scikit-learn scikit-image torch torchvision Pillow
```

## Data Augmentation
Because not all the characters have equal amounts of data, we used imagemorph  to augment the data. To compile and build imagemorph, change directories to the `Code` directory and run use the following command: `make` (in Windows use something like MinGW)
Then, to run the imagemorph program, use the following command:
```bash
./run_imagemorph.sh
```
This will ensure we have 300 examples for each class.


## How to use train and test our network:
Run `python3 main.py -h` to see all possible (and required) arguments. Please keep in mind that you should only have binarized images in your test/predict data folder. All of these images should also be JPGs.  

## Example uses
To train a network on the training set, we could use the command:
```bash
python3 main.py --train --train_dataroot path/to/train/data
```  

To use the network we just trained to predict the text in the unsegmented binarized images we could use the command:

```bash
python3 main.py path/to/test/data
```

## What is in the Code folder?
1. `segmentation.py` handles the line and character segmentation of input test images.
2. `preprocess.py` handles the cropping of the segmented characters. The characters are also centered in the resulting images.
3. `visualizer.py` handles intermediate results visualization (such as the images we use for debugging and presentations).
4. `characterclassifier.py` contains the code for our neural network. We are using a ResNet architecture.
5. `imagemorph.c` and `run_imagemorph.sh` handle the data augmentation, imagemorph.c has been edited slightly to allow for PGM inputs instead of PPM. 