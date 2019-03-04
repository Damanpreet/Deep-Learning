from __future__ import print_function
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bnorm2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bnorm4 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.4)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bnorm5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bnorm6 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.5)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.norm = nn.BatchNorm1d(1024) #Adding 1d batch normalization
        # self.fc_extra = nn.Linear(1024, 512)   #FC layer
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = self.bnorm1(x)
        x = F.elu(self.conv2(x))
        x = self.bnorm2(x)
        x = self.pool(x)
        x = self.dropout2(x)

        x = F.elu(self.conv3(x))
        x = self.bnorm3(x)
        x = F.elu(self.conv4(x))
        x = self.bnorm4(x)
        x = self.pool(x)
        x = self.dropout2(x)

        x = F.elu(self.conv5(x))
        x = self.bnorm5(x)
        x = F.elu(self.conv6(x))
        x = self.bnorm6(x)
        x = self.pool(x)
        x = self.dropout3(x)

        x = x.view(-1, self.num_flat_features(x))
       # x = F.relu(self.fc1(x))
        x = F.elu(self.norm(self.fc1(x))) #Adding Batch Norm here
        # x = F.elu(self.fc_extra(x))             #Adding a FC layer
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def eval_net(dataloader):
    correct = 0
    total = 0
    total_loss = 0
    net.eval() 
    criterion = nn.CrossEntropyLoss(size_average=False)
    for data in dataloader:
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        total_loss += loss.data[0]
    net.train() 
    return total_loss / total, correct / total

if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32          
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    MAX_EPOCH = 250 
    WEIGHT_DECAY = 1e-5

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)

    train_transform = transforms.Compose(
        [ transforms.RandomCrop(32, padding = 4),
          transforms.RandomHorizontalFlip(),
          transforms.ColorJitter(hue=.05, saturation=.05),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Augmenting the training dataset

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)
    print('Length of testing data: ', len(trainset))
    print('Length of training data:', len(testset))

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('Building model...')
    net = Net().cuda()
    net.train() 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay = WEIGHT_DECAY, momentum=MOMENTUM)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # scheduler.step()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 500 == 499:    # print every 2000 mini-batches
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
        print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(trainloader)
        test_loss, test_acc = eval_net(testloader)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))
        train_losses.append(train_loss)
        train_accuracies.append(train_acc*100)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc*100)

    print('Finished Training')
    print('Plotting graphs: ')
    
    # Saving accuracy and loss
    save_directory = './Model/AugmentedSGD/'
    import os
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    accuracy_loss = open(save_directory+"Output.txt", "w")
    accuracy_loss.write("Testing loss\n")
    for test_loss in test_losses:
        accuracy_loss.write("%s\n" % str(test_loss))

    accuracy_loss.write("\n\nTraining loss")
    for train_loss in train_losses:
        accuracy_loss.write("\n"+str(train_loss))
    
    accuracy_loss.write("\n\nTesting accuracy")
    for test_acc in test_accuracies:
        accuracy_loss.write("\n"+str(test_acc))
    
    accuracy_loss.write("\n\nTraining accuracy")
    for train_acc in train_accuracies:
        accuracy_loss.write("\n"+str(train_acc))
    accuracy_loss.close()

    # Plot the graph
    plt.figure()
    plt.xticks(np.arange(0, len(train_accuracies), 10))
    plt.xlabel('No of Epochs')
    plt.ylabel('Accuracy (in %)')
    plt.title('ACCURACY vs No of Epochs (SGD)\n  (LR: '+str(LEARNING_RATE)+', Momentum: '+str(MOMENTUM)+ ', Batch Size: '+str(BATCH_SIZE)+')')
    r, = plt.plot(train_accuracies, label='Training accuracy')
    b, = plt.plot(test_accuracies, label='Testing accuracy')
    plt.legend((r, b), "best")
    plt.savefig(save_directory+'accuracy.png')

    plt.figure()
    plt.xticks(np.arange(0, len(train_losses), 10))
    plt.xlabel('No of Epochs')
    plt.ylabel('Loss')
    plt.title('LOSS vs No of Epochs (SGD)\n  (LR: '+str(LEARNING_RATE)+' Momentum: '+str(MOMENTUM)+' Batch Size: '+str(BATCH_SIZE)+')')
    r, = plt.plot(train_losses, label='Training loss')
    b, = plt.plot(test_losses, label='Testing loss')
    plt.legend((r, b), "best")
    plt.savefig(save_directory+'loss.png')

    print('Saving model...')
    torch.save(net.state_dict(), save_directory+'mytraining.pth')
