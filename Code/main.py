"""main.py
This will be the file that serves as a starting point.
usage: python main.py [-h]
"""

import os
import argparse
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import numpy as np
import PIL.Image as Image
import util
import characterclassifier as cc
import segmentation
import preprocess
import json

def parse_args():
    """Parses command line arguments and defines defaults for
    user-defined variables."""

    parser = argparse.ArgumentParser(description="Character classifier for the Dead Sea Scrolls")
    # Base options
    parser.add_argument("--train_dataroot", required=True, type=str, help="root-directory containing the training set images. It is required to know what the target labels are.")
    test_img_data_group = parser.add_mutually_exclusive_group()
    test_img_data_group.add_argument("--test_image", type=str, help="filename of an image you want to predict")
    test_img_data_group.add_argument("--test_dataroot", type=str, help="root-directory containing the testing set images (unsegmented)")
    train_test_group = parser.add_mutually_exclusive_group(required=True)
    train_test_group.add_argument("-tr", "--train", action="store_true", help="whether to train a classifier")
    train_test_group.add_argument("-te", "--test", action= "store_true", help="whether to test a classifier")
    train_test_group.add_argument("-pr", "--predict", action="store_true", help="whether to use a classifier to predict the label of a character image")
    parser.add_argument("--n_channels", type=int, default=1, help="number of color channels in the input data")
    parser.add_argument("--image_size", type=int, default=64, metavar="SIZE", help="both width and height of input images will be scaled to be SIZE pixels large")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of gpus that can be used")

    # Network options
    parser.add_argument("--nf", type=int, default=16, help="number of feature maps in conv layers")
    parser.add_argument("--use_bias", action="store_true", default=True, help="whether to use biases in the neural network")
    parser.add_argument("--n_workers", type=int, default=2, help="number of workers for the dataloader")
    parser.add_argument("--n_classes", type=int, default=27, help="number of classes in the domain (letters in the Hebrew alphabet)")
    parser.add_argument("--n_resnet_blocks", type=int, default=10, help="number of resnet blocks in the neural network")

    # Train options
    parser.add_argument("-ne", "--n_epochs", type=int, default=15, help="number of training epochs")
    parser.add_argument("-b1", "--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("-b2", "--beta2", type=float, default=0.999, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="(initial) learning rate used in the optimizer")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="minibatch size")
    parser.add_argument("-sf", "--save_frequency", type=int, default=5, metavar="FREQ", help="save the network after every FREQ epochs")
    parser.add_argument("-sd", "--save_dir", type=str, default="../Networks", metavar="DIR", help="dirname where networks will be saved")
    
    # Test/segmentation options
    parser.add_argument("--network_path", type=str, help="The filepath of the network you want to test.")
    parser.add_argument("-v", "--visualize", action="store_true", help="Tell the program whether to visualize intermediate results. See visualizer.py")
    parser.add_argument("--CONST_C", type=int, default=-80, help="The constant C in the formula for D(n). See A* paper.")
    parser.add_argument("-s", "--subsampling", type=int, default=4, help="The subsampling factor of the test image prior to performing the A* algorithm.")
    parser.add_argument("--CONST_C_CHAR", type=int, default=-366, help="The constant C in the formula for D(n), used for segmenting characters. See A* paper.")
    parser.add_argument("-sc", "--subsampling_char", type=int, default=1, help="The subsampling factor of the segmented image prior to performing the A* algorithm (for characters).")
    parser.add_argument("-p", "--persistence_threshold", type=int, default=2, help="The persistence threshold for finding local extrema.")
    parser.add_argument("--n_black_pix_threshold", type=int, default=200, help="The minimum number of black pixels per character image.")
    parser.add_argument("--prediction_file", default="../predictions.json", help="The location where the predictions should be saved to.")
    return parser.parse_args()


def create_dataloaders(args):
    """Wraps the creation of the dataset objects and the dataloader
    objects. Uses a few variables from the options:
    args.train_dataroot, args.image_size, args.batch_size, args.n_workers.

    https://discuss.pytorch.org/t/how-to-do-a-stratified-split/62290 helped a lot.
    """
    # This manual seed ensures we always get the same train, val and test set
    torch.manual_seed(1337)
    np.random.seed(1337)
    dataset = dset.ImageFolder(root=args.train_dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize((args.image_size, args.image_size)),
                                   transforms.Grayscale(num_output_channels=1),
                                   transforms.ToTensor(),
                               ])
                               )

    train_idx, test_idx = train_test_split(
                              np.arange(len(dataset.targets)),
                              test_size=0.15,
                              shuffle=True,
                              stratify=dataset.targets
                          )

    # MIGHTDO: Equalize validation and test sets 
    # validation set is 15% of the train set now, whereas the test set is 15% of the whole dataset
    train_idx, valid_idx = train_test_split(
                              train_idx,
                              test_size=0.15
                          )
    print(train_idx)
    print(test_idx)
    print(valid_idx)
    exit()

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.n_workers)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.n_workers)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.n_workers)

    return train_loader, valid_loader, test_loader

def train(args, network, train_data, nll_loss, optimizer, device, valid_data):
    """Performs the training loop.
    Args:
        args         -- user-defined options
        train_data   -- DataLoader object that yields data in batches, used for training
        nll_loss     -- PyTorch NLLLoss object, computes losses
        optimizer    -- Optimizer used during training
        device       -- Either cpu or cuda, cuda provides faster training
        valid_data   -- DataLoader object that yields data in batches, used for validation
    """
    for epoch_i in range(1, args.n_epochs+1):
        print(f"Epoch {epoch_i} of {args.n_epochs}")

        # Get the data in batches
        for data, targets in train_data:
            # We perform our custom preprocessing
            data = data.numpy()
            data = preprocess.arrs_to_tensor(data, args.image_size)
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = network(data)
            # Compute negative log-likelihood loss (needs LogSoftmax as last layer in the network)
            loss = nll_loss(predictions, targets)
            loss.backward()
            optimizer.step()
    
        if epoch_i % args.save_frequency == 0 or epoch_i == args.n_epochs:
            # Save the model in the save_dir
            network_name = f"network_{str(epoch_i).zfill(2)}.pt"
            print(f"Saved network: {network_name}")
            torch.save(network, f"{args.save_dir}/{network_name}")

        # Validate on validation set
        with torch.no_grad():
            test(args, network, valid_data, nll_loss, device, prefix="Val")

def test(args, network, test_data, nll_loss, device, prefix="Test"):
    losses = []
    n_correct = 0
    n_preds = 0
    for data, targets in test_data:
        # We perform our custom preprocessing
        data = data.numpy()
        # data = Image.fromarray(data)
        data = preprocess.arrs_to_tensor(data, args.image_size)
        data = data.to(device)
        targets = targets.to(device)
        predictions = network(data)

        for pred_i, pred in enumerate(predictions):
            if torch.argmax(pred) == targets[pred_i]:
                n_correct += 1
        n_preds += len(predictions)

        loss = nll_loss(predictions, targets)
        losses.append(loss.detach().cpu().numpy())
    avg_loss = np.average(np.array(losses))
    accuracy = n_correct / n_preds
    print(f"{prefix} Accuracy: {accuracy:.3f},    {prefix} loss: {avg_loss:.3f}")

def predict(args, network, data, device, labels):
    returned_characters = []
    for char_img in data:
        char_img = resize(char_img, (args.image_size, args.image_size))
        char_img = char_img.reshape(1, 1, char_img.shape[0], char_img.shape[1])
        char_img /= 255
        char_img = torch.Tensor(char_img)
        char_img = char_img.to(device)
        pred = network(char_img)
        pred = int(torch.argmax(pred).cpu())
        for key, value in labels.items():
            if value == pred:
                # TODO: Check if this results in the correct order, might need to insert(0) instead
                returned_characters.append(key)
                break
    return returned_characters

def argument_error(message):
    print("ERROR: " + message)
    exit(-1)

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    util.makedirs(args.save_dir)

    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(args)


    if args.train:
        clf = cc.CharacterClassifier(args).to(device)
        nll_loss = torch.nn.NLLLoss().to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
        train(args, clf, train_dataloader, nll_loss, optimizer, device, valid_dataloader)

    elif args.test:
        if not args.network_path:
            argument_error("No network_path specified in command line arguments, which is required if you want to test.")

        clf = torch.load(args.network_path)
        clf.eval()
        nll_loss = torch.nn.NLLLoss().to(device)
        test(args, clf, test_dataloader, nll_loss, device)

    elif args.predict:
        if not args.test_dataroot and not args.test_image:
            argument_error("No test_dataroot or test_image specified in command line arguments, which is required if you want to predict")
        elif not args.network_path:
            argument_error("No network_path specified in command line arguments, which is required if you want to predict.")

        class_labels = dset.ImageFolder(root=args.train_dataroot).class_to_idx

        if args.test_dataroot:
            print("The test dataroot is expected to only contain binarized images."
                  "Please check if this is the case")

            preds = dict()
            test_filenames = os.listdir(args.test_dataroot)
            for f_idx, filename in enumerate(test_filenames):
                if not ".jpg" in filename and not ".jpeg" in filename:
                    argument_error(f"The file {args.test_dataroot}/{filename} is not a JPEG file.")

                char_segments = segmentation.segment_from_args(args, filename)
                char_segments = preprocess.preprocess_arrays(char_segments, filename, args.visualize)
                clf = torch.load(args.network_path)
                clf.eval()
                pred = predict(args, clf, char_segments, device, class_labels)
                pred = " ".join(pred)
                preds[filename] = pred
                print(pred)

            # Output the predictions to a file specified in the args
            with open(args.prediction_file, "w") as outfile:
                json.dump(preds, outfile, sort_keys=True, indent=4)

        elif args.test_image:
            img = Image.open(args.test_image)
            img = np.array(img)
            clf = torch.load(args.network_path)
            clf.eval()
            pred = predict(args, clf, [img], device, class_labels)
            print(pred)

if __name__ == "__main__":
    main()