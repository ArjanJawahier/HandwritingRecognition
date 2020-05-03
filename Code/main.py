"""main.py
This will be the file that serves as a starting point.
usage: python main.py [-h] -f FILE
"""

import argparse
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import PIL

import util
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
    parser.add_argument("-is", "--imagesize", type=int, default=32, metavar="SIZE", help="both width and height of input images will be scaled to be SIZE pixels large")
    parser.add_argument("--ngpu", type=int, default=1, help="number of gpus that can be used")
    parser.add_argument("--nworkers", type=int, default=2, help="number of workers for the dataloader")
    parser.add_argument("--nclasses", type=int, default=27, help="number of classes in the domain (letters in the Hebrew alphabet)")
    parser.add_argument("--num_resnet_blocks", type=int, default=6, help="number of resnet blocks in the neural network")

    # Network options
    parser.add_argument("--dropout", action="store_true", help="whether to use dropout")
    parser.add_argument("--nf", type=int, default=16, help="number of feature maps in conv layers")
    parser.add_argument("--usebias", action="store_true", default=True, help="whether to use biases in the neural network")

    # Train options
    parser.add_argument("-ne", "--nepochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("-pa", "--patience", type=int, default=0, help="patience for early stopping")
    parser.add_argument("-b1", "--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("-b2", "--beta2", type=float, default=0.999, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("-lr", "--learningrate", type=float, default=0.001, help="(initial) learning rate used in the optimizer")
    parser.add_argument("-bs", "--batchsize", type=int, default=64, help="minibatch size")
    parser.add_argument("-sf", "--savefrequency", type=int, default=5, metavar="FREQ", help="save the network after every FREQ epochs")
    parser.add_argument("-sd", "--savedir", type=str, default="../Networks", metavar="DIR", help="dirname where networks will be saved")
    # Test options

    return parser.parse_args()


def create_dataloaders(opt):
    """Wraps the creation of the dataset objects and the dataloader
    objects. Uses a few variables from the options:
    opt.dataroot, opt.imagesize, opt.batchsize, opt.nworkers.

    https://discuss.pytorch.org/t/how-to-do-a-stratified-split/62290 helped a lot.
    """

    # This manual seed ensures we always get the same train, val and test set
    torch.manual_seed(1337)
    np.random.seed(1337)
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imagesize),
                                   transforms.CenterCrop(opt.imagesize),
                                   transforms.Grayscale(num_output_channels=1),
                                   transforms.ToTensor(),
                               ]))

    train_idx, test_idx = train_test_split(
                              np.arange(len(dataset.targets)),
                              test_size=0.15,
                              shuffle=True,
                              stratify=dataset.targets
                          )

    train_idx, valid_idx = train_test_split(
                              train_idx,
                              test_size=0.15
                          )

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize, sampler=train_sampler, num_workers=opt.nworkers)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize, sampler=valid_sampler, num_workers=opt.nworkers)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize, sampler=test_sampler, num_workers=opt.nworkers)
    return train_loader, valid_loader, test_loader

def train(opt, network, dataloader, nll_loss, optimizer, device):
    """Performs the training loop.
    Args:
        opt         -- user-defined options
        dataloader  -- DataLoader object that yields data in batches
    """
    for epoch_i in range(1, opt.nepochs+1):
        # Get the data in batches
        print(f"Epoch {epoch_i} of {opt.nepochs}")
        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)
            # Reset the gradient
            optimizer.zero_grad()

            # Get network output (prediction)
            prediction = clf(data)
            # Compute negative log-likelihood loss (needs LogSoftmax as last layer in the network)
            loss = nll_loss(prediction, targets)

            # Compute gradients
            loss.backward()

            # Adjust weights
            optimizer.step()
    
        if epoch_i % opt.savefrequency == 0 or epoch_i == opt.nepochs:
            # Save the model in the savedir
            network_name = f"network_{str(epoch_i).zfill(2)}"
            print(network_name)
            torch.save(network, f"{opt.savedir}/{network_name}")

if __name__ == "__main__":
    opt = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    util.makedirs(opt.savedir)
    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(opt)


    if opt.train:
        clf = cc.CharacterClassifier(opt).to(device)
        nll_loss = torch.nn.NLLLoss().to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=opt.learningrate, betas=(opt.beta1, opt.beta2))
        train(opt, clf, train_dataloader, nll_loss, optimizer, device)