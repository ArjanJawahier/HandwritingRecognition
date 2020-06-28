"""network.py
This is the file with all the code about the neural network's architecture.
"""

import torch
import torch.nn as nn

class CharacterClassifier(nn.Module):
    """A classifier class that consists of ResNet blocks.
    
    inputs:
    args -- the options object that contains all user-defined option variables
    """
    def __init__(self, args):
        super(CharacterClassifier, self).__init__()
        torch.manual_seed(1337)

        self.args = args
        self.layers = [nn.ReflectionPad2d(1),
                       nn.Conv2d(
                           args.n_channels, args.nf, kernel_size=3, 
                           padding=0, bias=args.use_bias
                       ),
                       nn.BatchNorm2d(args.nf),
                       nn.ReLU()]

        mult = 1 
        for i in range(args.n_resnet_blocks):
            self.layers += [ResnetBlock(args, mult)]

        self.layers += [Flatten(),
                        nn.Linear(
                            args.nf*mult*(args.image_size**2),
                            args.n_classes
                        ),
                        nn.LogSoftmax(dim=1)]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        """The forward-pass function lets data flow forwards
        through the network."""
        return self.model(x)

class Flatten(nn.Module):
    """This 'layer' flattens the input tensor. It is used between conv and
    linear layers. This class was copied from an answer from:
    https://stackoverflow.com/questions/53953460/
    """
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class ResnetBlock(nn.Module):
    def __init__(self, args, mult):
        """Resnet blocks are good for a deep network.
        Resnet blocks are able to avoid the vanishing gradient problem by 
        sending their activation forwards through the network, skipping a layer.
        """
        super(ResnetBlock, self).__init__()
        self.args = args
        self.layers = [nn.ReflectionPad2d(1),  # 3x3 conv -> need padding of 1
                       nn.Conv2d(
                           args.nf * mult, args.nf * mult,
                           kernel_size=3, padding=0, bias=args.use_bias
                       ),
                       nn.BatchNorm2d(args.nf * mult),
                       nn.ReLU(),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(
                           args.nf * mult, args.nf * mult,
                           kernel_size=3, padding=0, bias=args.use_bias
                       ),
                       nn.BatchNorm2d(args.nf * mult)] 
        self.block = nn.Sequential(*self.layers)

    def forward(self, x):
        return x + self.block(x)