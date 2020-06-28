# Handwriting Recognition
Handwriting Recognition project

## Dependencies
We are using quite a number of dependencies. Use the following command to install them:  
```bash
pip3 install -r REQUIREMENTS.txt
```

## Data Augmentation
Because not all the characters have equal amounts of data and because there is not enough data, we used imagemorph to augment the already existing data. To compile and build imagemorph, change directories to the `Code` directory and run use the following command: `make` (in Windows use something like MinGW)
Then, to run the imagemorph program, use the following command:
```bash
./run_imagemorph.sh
```
This will ensure we have 1000 examples for each class.


## How to use train and test our network:
Run `python3 main.py -h` to see all possible (and required) arguments. Please keep in mind that you should only have binarized images in your test/predict data folder. All of these images should also be JPGs.  

## Example uses
To train a network on the training set, we could use the command:
```bash
python3 main.py --train --train_dataroot path/to/train/data path/to/test/data
```  
Sadly, the path/to/test/data is a required argument at the moment, due to the positional nature as laid out in the requirements.

To use the network we just trained to predict the text in the unsegmented binarized images we could use the command:

```bash
python3 main.py path/to/test/data
```

The above command will also perform style classification on the binarized images.  

Note: If you want to visualize intermediate results, use the option -v. The program has been known to crash sometimes, due to the use of threads. We have not fixed this problem as of yet.

## What is in the Code folder?
1. `segmentation.py` handles the line and character segmentation of input test images.
2. `preprocess.py` handles the cropping of the segmented characters. The characters are also centered in the resulting images.
3. `visualizer.py` handles intermediate results visualization (such as the images we use for debugging and presentations).
4. `characterclassifier.py` contains the code for our neural network. We are using a ResNet architecture.
5. `edge_hinge.py` is all about the Edge Hinge feature. Edge Hinge features of test data are compared with the averaged Style-Training data Edge Hinge features, which is stored in the file `style_data.pkl` in the root directory.
6. `imagemorph.c` and `run_imagemorph.sh` handle the data augmentation, imagemorph.c has been edited slightly to allow for PGM inputs instead of PPM. 