#!/usr/bin/env python3 
import os
import argparse
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

# import adversarial attacks from .py
from adversarial_attacks import fgsm_attack, pgd_attack

# import defense
from defense import mixup_data

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

valid_size = 1024 
batch_size = 32 

'''Basic neural network architecture (from pytorch doc).'''
class Net(nn.Module):

    model_file="models/default_model.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

    def __init__(self, defense: str = None):
        super().__init__()
        self.rse = (defense == "rse")
        self.noise_level = 0.2
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        noise1 = torch.randn(x.shape) * self.noise_level if self.rse else torch.zeros_like(x)
        x = self.pool(F.relu(self.conv1(x + noise1)))
        noise2 = torch.randn(x.shape) * self.noise_level if self.rse else torch.zeros_like(x)
        x = self.pool(F.relu(self.conv2(x + noise2)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

        
    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''        
        self.load(os.path.join(project_dir, Net.model_file))

def train_model(net, train_loader, pth_filename, num_epochs, args):
    '''
    Basic training function (from pytorch doc.)

    The adversarial_attack parameter chooses the adversarial method to be used in adversarial training: Possible values = 'fgsm' or 'pgd'
    '''
    adversarial_attack = args.attack
    defense = args.defense
    print("Starting training with {} adversarial attack".format(adversarial_attack) + " and {} defense".format(defense))

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            loss_adv = 0.0

            if (defense == 'mixup'):
                inputs = mixup_data(inputs, labels, lam=0.6, device=device) # lam = 0.6 used in the paper

            if (adversarial_attack in ['fgsm', 'pgd']):
                inputs.requires_grad = True  # enable gradient calculation

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (adversarial_attack in ['fgsm', 'pgd']):
                # Generate adversarial examples
                if (adversarial_attack == 'fgsm'):
                    inputs_adv = fgsm_attack(inputs, args.epsilon, inputs.grad)
                if (adversarial_attack == 'pgd'):
                    inputs_adv = pgd_attack(net, inputs, labels, args.epsilon, args.alpha, args.num_iter)

                # Forward pass and loss calculation with adversarial examples
                outputs_adv = net(inputs_adv)
                loss_adv = criterion(outputs_adv, labels)

                # Backward pass and optimizer update with adversarial examples for adversarial training
                optimizer.zero_grad()
                loss_adv.backward()
                optimizer.step()

            # print statistics
            running_loss += loss.item() + loss_adv
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))

def test_natural(net, test_loader, args):
    '''Basic testing function.'''

    correct = 0
    total = 0
    for i, data in enumerate(test_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)

        if args.attack in ['fgsm', 'pgd']:
            images.requires_grad = True  # enable gradient calculation

            # Forward pass
            outputs = net(images)
            loss = F.nll_loss(outputs, labels)

            # Backward pass to calculate gradients
            net.zero_grad()
            loss.backward()

            if args.attack == 'fgsm':
                images = fgsm_attack(images, args.epsilon, images.grad)
            elif args.attack == 'pgd':
                images = pgd_attack(net, images, labels, args.epsilon, args.alpha, args.num_iter)

        with torch.no_grad():
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def test_rse(net, test_loader, args):
    correct = 0
    total = 0
    for i, data in enumerate(test_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)

        if args.attack in ['fgsm', 'pgd']:
            images.requires_grad = True  # enable gradient calculation

            # Forward pass
            outputs = net(images)
            loss = F.nll_loss(outputs, labels)

            # Backward pass to calculate gradients
            net.zero_grad()
            loss.backward()

            if args.attack == 'fgsm':
                images = fgsm_attack(images, args.epsilon, images.grad)
            elif args.attack == 'pgd':
                images = pgd_attack(net, images, labels, args.epsilon, args.alpha, args.num_iter)

        with torch.no_grad():
            # ensemble outputs
            bs = images.shape[0]
            outputs = torch.zeros(bs, 10).to(device)
            for i in range(50):
                outputs += net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def get_train_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the training part.'''

    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    train = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    return train

def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)

    return valid

def main():

    #### Parse command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=Net.model_file,
                        help="Name of the file used to load or to sore the model weights."\
                        "If the file exists, the weights will be load from it."\
                        "If the file doesn't exists, or if --force-train is set, training will be performed, "\
                        "and the model weights will be stored in this file."\
                        "Warning: "+Net.model_file+" will be used for testing (see load_for_testing()).")
    parser.add_argument('-f', '--force-train', action="store_true",
                        help="Force training even if model file already exists"\
                             "Warning: previous model file will be erased!).")
    parser.add_argument('-e', '--num-epochs', type=int, default=10,
                        help="Set the number of epochs during training")
    
    # add adversarial attack arguments
    parser.add_argument('-a', '--attack', type=str, default=None, choices=['fgsm', 'pgd'], 
                        help="Specify which adversarial attack to use")
    parser.add_argument('--epsilon', type=float, default=0.03,
                        help="Epsilon value for adversarial attack.")
    parser.add_argument('--alpha', type=float, default=0.01,
                        help="Alpha value for PGD attack.")
    parser.add_argument('--num-iter', type=int, default=40,
                        help="Number of iterations for PGD attack.")

    # select defense
    parser.add_argument('-d', '--defense', type=str, default=None, choices=['mixup', 'rse'],
                        help="Specify which defense to use")


    args = parser.parse_args()

    #### Create model and move it to whatever device is available (gpu/cpu)
    net = Net(args.defense)
    net.to(device)

    #### Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model")
        print(args.model_file)

        train_transform = transforms.Compose([transforms.ToTensor()]) 
        cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)

        train_model(net, train_loader, args.model_file, args.num_epochs, args)
        print("Model save to '{}'.".format(args.model_file))

    #### Model testing
    print("Testing with model from '{}'. ".format(args.model_file))

    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor()) 
    valid_loader = get_validation_loader(cifar, valid_size)

    net.load(args.model_file)
    original_attack = args.attack

    # ----------------------- args.attack is corrupted in this interval
    args.attack = None
    nat_acc = test_natural(net, valid_loader, args)
    print("Model natural accuracy (valid): {}".format(nat_acc))
    if args.defense == "rse":
        rse_acc = test_rse(net, valid_loader, args)
        print("              with RSE (valid): {}".format(rse_acc))

    args.attack = 'fgsm'
    fgsm_acc = test_natural(net, valid_loader, args)
    print("Model FGSM accuracy (valid): {}".format(fgsm_acc))
    if args.defense == "rse":
        rse_acc = test_rse(net, valid_loader, args)
        print("           with RSE (valid): {}".format(rse_acc))

    args.attack = 'pgd'
    pgd_acc = test_natural(net, valid_loader, args)
    print("Model PGD accuracy (valid): {}".format(pgd_acc))
    if args.defense == "rse":
        rse_acc = test_rse(net, valid_loader, args)
        print("          with RSE (valid): {}".format(rse_acc))
    args.attack = original_attack
    # -----------------------


    if args.model_file != Net.model_file:
        print("Warning: '{0}' is not the default model file, "\
              "it will not be the one used for testing your project. "\
              "If this is your best model, "\
              "you should rename/link '{0}' to '{1}'.".format(args.model_file, Net.model_file))

if __name__ == "__main__":
    main()

