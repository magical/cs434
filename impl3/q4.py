import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)

epochs = 20
batch_size = 32
learning_rate = 0.01

if len(sys.argv) > 1:
    learning_rate = float(sys.argv[1])

core_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
train_dataset = torch.utils.data.Subset(core_dataset, range(0, 40000))
validation_dataset = torch.utils.data.Subset(core_dataset, range(40000, 50000))
#test_dataset = datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
#test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

print(model)

def train(epoch, log_interval=200):
    # Set model to training mode
    model.train()

    lossv = []

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
        lossv.append(loss.data.item())

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

    return sum(lossv) / len(lossv)

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
    accuracy_vector.append(accuracy)
    
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))


model_dir = "q4-model"
try:
    os.mkdir(model_dir)
except FileExistsError:
    pass

training_lossv = []
lossv, accv = [], []
for epoch in range(1, epochs + 1):
    loss = train(epoch)
    training_lossv.append(loss)
    print()
    print("Average loss:", loss)
    validate(lossv, accv)
    print()

    filename = os.path.join(model_dir, "q4-lr{0:f}-epoch{1:d}".format(learning_rate, epoch))
    torch.save(model.state_dict(), filename)


datafilename = os.path.join(model_dir, "q4-lr{0}.data".format(learning_rate, epoch))
with open(datafilename, "w") as datafile:
    print("loss", *training_lossv, sep="\t", file=datafile)
    print("accuracy", *accv, sep="\t", file=datafile)
