import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm


# Note the model and functions here defined do not have any FL-specific components.


class Net0(nn.Module):
    """A simple CNN suitable for simple vision tasks."""

    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# This is Net_1
class Net(nn.Module):
    def __init__(self, num_classes, channels):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the feature maps
        x = x.view(-1, 128 * 3 * 3)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# This is Net_2
class Net2(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 512, kernel_size=3, padding=1)

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, num_classes)  # 10 output classes for CIFAR-10

    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        # Flatten the feature maps
        x = x.view(-1, 512)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(weights=None)
        # self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(weights=None)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)


class VGG11(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(VGG11, self).__init__()

        self.conv_block1 = self._make_conv_block(1, 64, 2)
        self.conv_block2 = self._make_conv_block(64, 128, 2)
        self.conv_block3 = self._make_conv_block(128, 256, 3)
        self.conv_block4 = self._make_conv_block(256, 512, 3)
        self.conv_block5 = self._make_conv_block(512, 512, 3)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        # dropout
        self.dropout = nn.Dropout(p=.5)
    def _make_conv_block(self, in_channels, out_channels, num_convs):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))  # BatchNorm after the convolution
        layers.append(nn.ReLU())
        for _ in range(num_convs - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))  # BatchNorm after the convolution
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)

        x = self.avg_pool(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class VGG111(nn.Module):
    def __init__(self, num_classes):
        super(VGG11, self).__init__()
        self.in_channels = 1
        self.num_classes = num_classes
        # convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # fully connected linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # flatten to prepare for the fully connected layers
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def train(client, dataset_name, device, run_round=0):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    print(f'Training client: {client.client_id} in round: {run_round}')

    criterion = torch.nn.CrossEntropyLoss()
    client.model.train()
    client.model.to(device)
    train_losses = {}
    # if run_round < 15:
    #     optimizer = optim.SGD(client.model.parameters(), lr=0.01, momentum=0.9)
    # else:
    optimizer = optim.Adam(client.model.parameters(), lr=0.001)

    for epoch in range(client.local_epochs):
        train_losses = {}
        running_loss = 0.0
        for images, labels in tqdm(client.train_loader, desc='Training'):
            if dataset_name == 'MNIST' or dataset_name == "CIFAR10":
                labels = labels.to(device)
            else:
                labels = torch.squeeze(labels, 1).long().to(device)
            images= images.to(device)
            optimizer.zero_grad()
            loss = criterion(client.model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # print(f'[{epoch}] loss: {running_loss}')
        train_losses[epoch] = running_loss
    # return client.model.state_dict(), train_losses
    print('Finished Training')
    return train_losses


def test(client, dataset_name, device):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    client.model.eval()
    client.model.to(device)
    with torch.no_grad():
        # for data in client.validation_loader:
        for images, labels in tqdm(client.validation_loader, desc='Local_testing', colour='yellow'):

            if dataset_name == 'MNIST' or dataset_name == "CIFAR10":
                labels = labels.to(device)
            else:
                labels = torch.squeeze(labels, 1).long().to(device)

            images = images.to(device)
            outputs = client.model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(client.validation_loader.dataset)
    return loss, accuracy


def test_global(glb_model, dataset_name, test_loader, device, last_round = False):
    print('Evaluating the global model')
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    glb_model.eval()
    glb_model.to(device)

    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        # for data in test_loader:
        for images, labels in tqdm(test_loader, desc='Testing', colour='green'):

            if dataset_name == 'MNIST' or dataset_name == "CIFAR10":
                labels = labels.to(device)
            else:
                labels = torch.squeeze(labels, 1).long().to(device)
            images = images.to(device)
            outputs = glb_model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            if last_round:
                # Convert predictions and labels to numpy arrays and store them
                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())

    accuracy = correct / len(test_loader.dataset)

    if last_round:
        # Convert predictions and labels to numpy arrays
        import numpy as np
        all_predictions = np.array(all_predictions)
        all_true_labels = np.array(all_true_labels)
        return loss, accuracy, all_predictions, all_true_labels

    return loss, accuracy
