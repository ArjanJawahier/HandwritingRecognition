"""network.py
This is the file with all the code about the neural network's architecture.
"""

import torch
import torch.nn as nn


class CharacterClassifier(nn.Module):
    """A classifier class that consists of ResNet blocks.
    
    Args:
        opt -- the options object that contains all user-defined option variables
    """
    def __init__(self, opt):
        super(CharacterClassifier, self).__init__()
        self.opt = opt
        self.layers = [nn.ReflectionPad2d(1),
                       nn.Conv2d(opt.nchannels, opt.nf, kernel_size=3, padding=0, bias=opt.usebias),
                       nn.BatchNorm2d(opt.nf),
                       nn.ReLU()]

        mult = 1 
        for i in range(opt.num_resnet_blocks):
            self.layers += [ResnetBlock(opt, mult)]

        self.layers += [Flatten(),
                        nn.Linear(opt.nf*mult*(opt.imagesize**2), opt.nclasses),
                        nn.LogSoftmax(dim=1)]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        """The forward-pass function lets data flow forwards
        through the network."""
        return self.model(x)

class Flatten(nn.Module):
    """This 'layer' flattens the input tensor.
    It is used between conv and linear layers.
    https://stackoverflow.com/questions/53953460/how-to-flatten-input-in-nn-sequential-in-pytorch
    """
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class ResnetBlock(nn.Module):
    def __init__(self, opt, mult):
        """Resnet blocks are good for a deep network.
        Resnet blocks are able to avoid the vanishing/exploding gradient problem by 
        sending their activation forwards through the network, skipping a layer.
        """
        super(ResnetBlock, self).__init__()
        self.opt = opt
        self.layers = [nn.ReflectionPad2d(1),  # The value is 1 because we use 3x3 convolutions
                       nn.Conv2d(opt.nf * mult, opt.nf * mult, kernel_size=3, padding=0, bias=opt.usebias),
                       nn.BatchNorm2d(opt.nf * mult),
                       nn.ReLU(),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(opt.nf * mult, opt.nf * mult, kernel_size=3, padding=0, bias=opt.usebias),
                       nn.BatchNorm2d(opt.nf * mult)] 
        self.block = nn.Sequential(*self.layers)

    def forward(self, x):
        return x + self.block(x)