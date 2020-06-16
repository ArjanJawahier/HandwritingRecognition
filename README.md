# Handwriting Recognition
Handwriting Recognition project

## Dependencies
We are using quite a number of dependencies, use the following command to install them:  
```bash
pip3 install numpy scikit-learn scikit-image torch torchvision Pillow
```

## Data Augmentation
Because not all the characters have equal amounts of data, we used imagemorph  to augment the data. To compile and build imagemorph, change directories to the `Code`directory and run use the following command: `make`  
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
python3 main.py --train --train_dataroot ../Train_Data/ 
```  

To use the network we just trained to predict the text in the unsegmented binarized images we could use the command:

```bash
python3 main.py --predict --test_dataroot ../Test_Data --network_path ../Networks/network_10.pt  --train_dataroot ../Train_Data/ -v
```
It is still necessary to specify a train_dataroot while predicting, because we can get the labels from this directory.

## What is in the Code folder?
1) `main.py` is the only file that should be directly run. It uses all the other code in the codebase to train a model and predict the letters in unsegmented, binarized test images.
2) `segmentation.py` handles the line and character segmentation of input test images.
3) `preprocess.py` handles the cropping of the segmented characters. The characters are also centered in the resulting images.
4) `visualizer.py` handles intermediate results visualization (such as the images we use in the presentations).
5) `characterclassifier.py` contains the code for our neural network. We are using a ResNet architecture.
6) `imagemorph.c` and `run_imagemorph.sh` handle the data augmentation, imagemorph.c has been edited slightly to allow for PGM inputs instead of PPM. 