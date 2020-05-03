"""main.py
This will be the file that serves as a starting point.
usage: python main.py [-h] -f FILE
"""

import argparse
import torch
import torch.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

import characterclassifier as cc

def parse_args():
	"""Parses command line arguments and defines defaults for
	user-defined variables."""

	parser = argparse.ArgumentParser(description="Character classifier for the Dead Sea Scrolls")
	# Base options
	parser.add_argument("-dr", "--dataroot", type=str, required=True, help="root-directory containing the images")
	train_test_group = parser.add_mutually_exclusive_group(required=True)
	train_test_group.add_argument("-tr", "--train", action="store_true", help="whether to train a classifier")
	train_test_group.add_argument("-te", "--test", action= "store_true", help="whether to test a classifier")
	train_test_group.add_argument("-pr", "--predict", action="store_true", help="whether to use a classifier to predict the label of a character image")
	parser.add_argument("--nchannels", type=int, default=1, help="number of color channels in the input data")
	parser.add_argument("-is", "--imagesize", type=int, default=64, metavar="SIZE", help="both width and height of input images will be scaled to be SIZE pixels large")
	parser.add_argument("--ngpu", type=int, default=1, help="number of gpus that can be used")
	parser.add_argument("--nworkers", type=int, default=2, help="number of workers for the dataloader")
	parser.add_argument("--nclasses", type=int, default=27, help="number of classes in the domain (letters in the Hebrew alphabet)")

	# Network options
	parser.add_argument("--dropout", action="store_true", help="whether to use dropout")
	parser.add_argument("--nf", type=int, default=64, help="number of feature maps in conv layers")

	# Train options
	parser.add_argument("-ne", "--nepochs", type=int, default=20, help="number of training epochs")
	parser.add_argument("-pa", "--patience", type=int, default=0, help="patience for early stopping")
	parser.add_argument("-b1", "--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
	parser.add_argument("-b2", "--beta2", type=float, default=0.999, help="beta2 parameter for the Adam optimizer")
	parser.add_argument("-lr", "--learningrate", type=float, default=0.001, help="(initial) learning rate used in the optimizer")
	parser.add_argument("-bs", "--batchsize", type=int, default=64, help="minibatch size")
	parser.add_argument("-sf", "--savefrequency", type=int, default=5, metavar="FREQ", help="save the network after every FREQ epochs")
	parser.add_argument("-sd", "--savedir", type=str, default="Networks", metavar="DIR", help="dirname where networks will be saved")
	# Test options

	return parser.parse_args()


def create_dataloader(opt):
	"""Wraps the creation of the dataset object and the dataloader
	object. Uses a few variables from the options:
	opt.dataroot, opt.imagesize, opt.batchsize, opt.nworkers."""

	dataset = dset.ImageFolder(root=opt.dataroot,
	                           transform=transforms.Compose([
	                               transforms.Resize(opt.imagesize),
	                               transforms.CenterCrop(opt.imagesize),
	                               transforms.ToTensor(),
	                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	                           ]))

	# Create the dataloader
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize,
	                                         shuffle=True, num_workers=opt.nworkers)
	return dataloader

def train(opt, dataloader):
	"""Performs the training loop.
	Args:
		opt 		-- user-defined options
		dataloader  -- DataLoader object that yields data in batches
	"""
	for epoch_i in range(opt.nepochs):
		# Get the data in batches
		for data_i, data in dataloader:
			print(data.shape)

if __name__ == "__main__":
	opt = parse_args()
	if opt.train:
		dataloader = create_dataloader(opt)
		clf = cc.CharacterClassifier(opt)
		print(clf)
		train(opt, dataloader)