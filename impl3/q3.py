import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
#import seaborn as sns
#sns.set()

#import numpy as np
#import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)

batch_size = 32
learning_rate = 0.01

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", dest="learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("dropout", type=float, default=0.0)
    parser.add_argument("momentum", type=float, default=0.0)
    parser.add_argument("weight_decay", type=float, default=0.0)
    return parser.parse_args()

args = parse_args()
print("arguments: ", sys.argv, args)

# DONE: scale input features? idk, seems to already be done
# DONE: dump model parameters after each epoch
# TODO: plots

# output: (data for) plots of training loss and validation error as a function of training epoch,
# for a variety of learning rates

core_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
train_dataset = torch.utils.data.Subset(core_dataset, range(0, 40000))
validation_dataset = torch.utils.data.Subset(core_dataset, range(40000, 50000))
test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

#pltsize=1
#plt.figure(figsize=(10*pltsize, pltsize))
#
#for i in range(10):
#    plt.subplot(1,10,i+1)
#    plt.axis('off')
#    plt.imshow(X_train[i,:,:,:].numpy().reshape(28,28), cmap="gray")
#    plt.title('Class: '+str(y_train[i].item()))

class Net(nn.Module):
    def __init__(self, dropout=0.2):
        super(Net, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc1_drop = nn.Dropout(dropout)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

# TODO: play around with learning rate
model = Net(dropout=args.dropout).to(device)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)
criterion = nn.CrossEntropyLoss()

print(model)

def train(epoch, log_interval=200):
    # Set model to training mode
    model.train()

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad()

        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))



def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(float(accuracy))

    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))


model_dir = "q3-model"
try:
    os.mkdir(model_dir)
except FileExistsError:
    pass

lossv, accv = [[],[]], [[],[]]
for epoch in range(1, args.epochs + 1):
    train(epoch)
    print()
    validate(lossv[0], accv[0])
    #test(lossv[1],accv[1])
    print()

    filename = os.path.join(model_dir, "q3-d{0.dropout}-m{0.momentum}-wd{0.weight_decay}-epoch{1:d}".format(args, epoch))
    torch.save(model.state_dict(), filename)

print(lossv)
print(accv)

#plt.figure(figsize=(5,3))
#plt.plot(np.arange(1,epochs+1), lossv)
#plt.title('validation loss')
#
#plt.figure(figsize=(5,3))
#plt.plot(np.arange(1,epochs+1), accv)
#plt.title('validation accuracy');
