"""main.py
This will be the file that serves as a starting point.
usage: python main.py [-h]
"""

import sys
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
import matplotlib.pyplot as plt

import Code.util as util
import Code.characterclassifier as cc
import Code.segmentation as segmentation
import Code.edge_hinge as edge_hinge
import Code.preprocess as preprocess

def parse_args():
    """Parses command line arguments and defines defaults for
    user-defined variables."""

    parser = argparse.ArgumentParser(description="Character classifier for the Dead Sea Scrolls")
    # Base options
    parser.add_argument("test_dataroot", type=str, default="Test_Data/", help="root-directory containing the testing set images (unsegmented)")
    parser.add_argument("--train_dataroot", type=str, help="root-directory containing the training set images. It is required to know what the target labels are.")
    train_test_group = parser.add_mutually_exclusive_group()
    train_test_group.add_argument("-tr", "--train", action="store_true", help="whether to train a classifier, if both train and test are not specified, we predict the contents of the test_dataroot")
    train_test_group.add_argument("-te", "--test", action= "store_true", help="whether to test a classifier, if both train and test are not specified, we predict the contents of the test_dataroot")
    parser.add_argument("--n_channels", type=int, default=1, help="number of color channels in the input data")
    parser.add_argument("--image_size", type=int, default=64, metavar="SIZE", help="both width and height of input images will be scaled to be SIZE pixels large")

    # Network options
    parser.add_argument("--nf", type=int, default=16, help="number of feature maps in conv layers")
    parser.add_argument("--use_bias", action="store_true", default=True, help="whether to use biases in the neural network")
    parser.add_argument("--n_workers", type=int, default=2, help="number of workers for the dataloader")
    parser.add_argument("--n_classes", type=int, default=27, help="number of classes in the domain (letters in the Hebrew alphabet)")
    parser.add_argument("--n_resnet_blocks", type=int, default=6, help="number of resnet blocks in the neural network")

    # Train options
    parser.add_argument("-ne", "--n_epochs", type=int, default=10, help="number of training epochs")
    parser.add_argument("-b1", "--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("-b2", "--beta2", type=float, default=0.999, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="(initial) learning rate used in the optimizer")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="minibatch size")
    parser.add_argument("-sf", "--save_frequency", type=int, default=5, metavar="FREQ", help="save the network after every FREQ epochs")
    parser.add_argument("-sd", "--save_dir", type=str, default="Networks", metavar="DIR", help="dirname where networks will be saved")
    
    # Test/segmentation options
    parser.add_argument("--network_path", type=str, default="Networks/final_network_05_epochs_1000_6_blocks.pt", help="The filepath of the network you want to test.")
    parser.add_argument("-v", "--visualize", action="store_true", help="Tell the program whether to visualize intermediate results. See visualizer.py")
    parser.add_argument("--CONST_C", type=int, default=-80, help="The constant C in the formula for D(n). See A* paper.")
    parser.add_argument("-s", "--subsampling", type=int, default=4, help="The subsampling factor of the test image prior to performing the A* algorithm.")
    parser.add_argument("--CONST_C_CHAR", type=int, default=-366, help="The constant C in the formula for D(n), used for segmenting characters. See A* paper.")
    parser.add_argument("-sc", "--subsampling_char", type=int, default=1, help="The subsampling factor of the segmented image prior to performing the A* algorithm (for characters).")
    parser.add_argument("-p", "--persistence_threshold", type=int, default=1, help="The persistence threshold for finding local extrema.")
    parser.add_argument("--n_black_pix_threshold", type=int, default=200, help="The minimum number of black pixels per character image.")
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
    train_accs, train_losses, val_accs, val_losses = [], [], [], []

    for epoch_i in range(1, args.n_epochs+1):
        print(f"Epoch {epoch_i} of {args.n_epochs}")

        # Get the data in batches
        batch_index = 0
        for data, targets in train_data:
            # We perform our custom preprocessing
            print(f"Batch {batch_index}/{len(train_data.dataset)/len(data)}", end="\r")
            data = data.numpy()
            data = preprocess.arrs_to_tensor(data, args)
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = network(data)
            # Compute negative log-likelihood loss (needs LogSoftmax as last layer in the network)
            loss = nll_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            batch_index += 1

        if epoch_i % args.save_frequency == 0 or epoch_i == args.n_epochs:
            # Save the model in the save_dir
            network_name = f"network_{str(epoch_i).zfill(2)}.pt"
            print(f"Saved network: {network_name}")
            torch.save(network.state_dict(), f"{args.save_dir}/{network_name}")

        # Validate on validation set
        with torch.no_grad():
            train_acc, train_loss = test(args, network, train_data, nll_loss, device)
            val_acc, val_loss = test(args, network, valid_data, nll_loss, device)
            print(f"Train acc: {train_acc:.3f}, Train loss: {train_loss:.3f}")
            print(f"Val acc: {val_acc:.3f}, Val loss: {val_loss:.3f}")
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            val_accs.append(val_acc)
            val_losses.append(val_loss)

    return train_accs, train_losses, val_accs, val_losses


def test(args, network, test_data, nll_loss, device):
    losses = []
    n_correct = 0
    n_preds = 0
    for data, targets in test_data:
        # We perform our custom preprocessing
        data = data.numpy()
        data = preprocess.arrs_to_tensor(data, args)
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
    return accuracy, avg_loss

def predict(args, network, data, device, labels):
    returned_characters = []
    returned_labels = []
    for char_img in data:
        char_img = resize(char_img, (args.image_size, args.image_size))
        char_img = char_img.reshape(1, 1, char_img.shape[0], char_img.shape[1])
        char_img = torch.Tensor(char_img)
        char_img = char_img.to(device)
        pred = network(char_img)
        pred = int(torch.argmax(pred).cpu())
        for key, value in labels.items():
            if value == pred:
                returned_characters.append(key)
                returned_labels.append(value)
                break
    return_array = np.array([returned_characters,returned_labels])
    return return_array

def argument_error(message):
    print("ERROR: " + message)
    exit(-1)

def most_frequent(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num 

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.train:
        train_dataloader, valid_dataloader, _ = create_dataloaders(args)

        util.makedirs(args.save_dir)
        clf = cc.CharacterClassifier(args).to(device)
        nll_loss = torch.nn.NLLLoss().to(device)
        optimizer = torch.optim.Adam(
            clf.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2)
        )
        train_accs, train_losses, val_accs, val_losses= train(
            args, clf, train_dataloader, nll_loss,
            optimizer, device, valid_dataloader
        )
        fig = plt.figure()
        plt.plot(train_accs, 'b')
        plt.plot(val_accs, 'r')
        plt.title("Train accuracy vs. validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(["Train", "Val"])
        plt.savefig("acc_curves.pdf")

        fig = plt.figure()
        plt.plot(train_losses, 'b')
        plt.plot(val_losses, 'r')
        plt.title("Train loss vs. validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Negative log-likelihood loss")
        plt.legend(["Train", "Val"])
        plt.savefig("loss_curves.pdf")


    elif args.test:
        _, _, test_dataloader = create_dataloaders(args)

        if not args.network_path:
            argument_error(
                "No network_path specified in command line arguments, "
                "which is required if you want to test."
            )

        clf = cc.CharacterClassifier(args).to(device)
        if device == torch.device("cuda:0"):
            clf.load_state_dict(torch.load(args.network_path))
        else:
            clf.load_state_dict(torch.load(args.network_path, map_location=lambda storage, loc: storage))

        clf.eval()
        nll_loss = torch.nn.NLLLoss().to(device)
        test_acc, test_loss = test(args, clf, test_dataloader, nll_loss, device)
        print(f"Test acc: {test_acc:.3f}, test loss: {test_loss:.3f}")

    else:
        if not os.path.isdir(args.test_dataroot):
            argument_error(
                f"The specified test_dataroot {args.test_dataroot} "
                "is not a directory!"
            )
        elif not args.network_path:
            argument_error(
                "No network_path specified in command line arguments, "
                "which is required if you want to predict."
            )

        class_labels = {
            'Alef': 0, 'Ayin': 1, 'Bet': 2, 'Dalet': 3, 'Gimel': 4, 'He': 5,
            'Het': 6, 'Kaf': 7, 'Kaf-final': 8, 'Lamed': 9, 'Mem': 10, 
            'Mem-medial': 11, 'Nun-final': 12, 'Nun-medial': 13, 'Pe': 14,
            'Pe-final': 15, 'Qof': 16, 'Resh': 17, 'Samekh': 18, 'Shin': 19,
            'Taw': 20, 'Tet': 21, 'Tsadi-final': 22, 'Tsadi-medial': 23, 
            'Waw': 24, 'Yod': 25, 'Zayin': 26
        }

        # These are the unicodes for the Hebrew characters
        unicode_labels = {
            '\u05D0': 0, '\u05E2': 1, '\u05D1': 2, '\u05D3': 3, '\u05D2': 4,
            '\u05D4': 5, '\u05D7': 6, '\u05DB': 7, '\u05DA': 8, '\u05DC': 9,
            '\u05DD': 10, '\u05DE': 11, '\u05DF': 12, '\u05E0': 13, '\u05E4': 14,
            '\u05E3': 15, '\u05E7': 16, '\u05E8': 17, '\u05E1': 18, '\u05E9': 19,
            '\u05EA': 20, '\u05D8': 21, '\u05E5': 22, '\u05E6': 23, '\u05D5': 24,
            '\u05D9': 25, '\u05D6': 26
        }

        style_classifier = edge_hinge.StyleClassifier("../Style_Data/")

        print("The test dataroot is expected to only contain binarized "
              "images. Please check if this is the case.")

        util.makedirs("results")
        clf = cc.CharacterClassifier(args).to(device)
        if device == torch.device("cuda:0"):
            clf.load_state_dict(torch.load(args.network_path))
        else:
            clf.load_state_dict(torch.load(args.network_path, map_location=lambda storage, loc: storage))

        clf.eval()

        test_filenames = os.listdir(args.test_dataroot)
        for f_idx, filename in enumerate(test_filenames):
            name, ext = os.path.splitext(filename)
            if ext != ".jpg" and ext != ".jpeg":
                print(f"The file {args.test_dataroot}/{filename} is not a JPEG file.")
                continue

            char_segments = segmentation.segment_from_args(args, filename)
            char_segments = preprocess.preprocess_arrays(char_segments, args, filename)
            pred_lines = []
            pred_styles = []
            for line in char_segments:
                # Do we use Class or Unicode labels?
                pred = predict(args, clf, line, device, unicode_labels)
                pred_uni = pred[0]
                pred_lab = pred[1]
                if len(pred_uni) > 0:
                    pred_uni = " ".join(list(reversed(pred_uni))) + "\n"
                    pred_lines.append(pred_uni)
                    for key, char in enumerate(line):
                        labelled_character = "none"
                        for key2, value in class_labels.items():
                            if int(value) == int(pred_lab[key]):
                                labelled_character = key2
                        style = style_classifier.predict_style(Image.fromarray(char*255), labelled_character)
                        pred_styles.append(style)

            with open(f"results/{name}_characters.txt", "w", encoding="utf-8") as outfile:
                outfile.writelines(pred_lines)
                print(f"Written output to results/{name}_characters.txt")

            print("=========")
            print(pred_styles)
            print("=========")
            with open(f"results/{name}_style.txt", "w", encoding="utf-8") as outfile:
                outfile.writelines(most_frequent(pred_styles))
                print(f"classified style: ", most_frequent(pred_styles), "\n")

if __name__ == "__main__":
    main()