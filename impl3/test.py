import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
#import seaborn as sns
#sns.set()

import numpy as np
#import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)

batch_size = 32
test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor()) 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def test(model, loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(test_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(test_loader.dataset)
    accuracy_vector.append(float(accuracy))
    
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        val_loss, correct, len(test_loader.dataset), accuracy))

class Q1(nn.Module):
    def __init__(self):
        super(Q1, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = torch.sigmoid(self.fc1(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

class Q2(nn.Module):
    def __init__(self):
        super(Q2, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

class Q3(nn.Module):
    def __init__(self, dropout=0.2):
        super(Q3, self).__init__()
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

class Q4(nn.Module):
    def __init__(self):
        super(Q4, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

criterion = nn.CrossEntropyLoss()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dropout", type=float, default=None)
    parser.add_argument("filename", type=str)
    args = parser.parse_args()
    if "q1" in args.filename:
        print("Using Q1 network")
        model = Q1().to(device)
    elif "q2" in args.filename:
        print("Using Q2 network")
        model = Q2().to(device)
    elif "q3" in args.filename:
        print("Using Q3 network")
        if args.dropout is None:
            print("error: must specify --dropout for q3 models", file=sys.stderr)
            sys.exit(1)
        model = Q3().to(device)
    elif "q4" in args.filename:
        print("Using Q4 network")
        model = Q4().to(device)
    else:
        print("error: couldn't guess which network to use", file=sys.stderr)
        sys.exit(1)

    model.load_state_dict(torch.load(args.filename))
    print(model)
    test(model, [],[])

main()
