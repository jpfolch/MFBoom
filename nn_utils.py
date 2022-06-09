from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader

class CNN(nn.Module):
    def __init__(self, hidden_layer_size = 100, learning_rate = 0.0001, num_of_epochs = 25):
        super().__init__()
        # network hyper-parametes
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.num_of_epochs = num_of_epochs
        # define the layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, self.hidden_layer_size)
        self.fc2 = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.fc3 = nn.Linear(self.hidden_layer_size, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train(self, X_train, Y_train, batch_size = 256, verbose = True):
        # transform data to tensor
        tensor_x = torch.Tensor(X_train)
        tensor_y = torch.Tensor(Y_train)
        tensor_y = tensor_y.type(torch.LongTensor)
        # create your datset
        dataset = TensorDataset(tensor_x,tensor_y) 
        # define loader
        trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # define criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)

        for epoch in range(self.num_of_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if (i % 10 == 9) & (verbose == True):    # print every 10 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                    running_loss = 0.0

        if verbose:
            print('Finished Training')