import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms



# Function to normalize the images
f = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the training data
trainset = datasets.MNIST(root='./../data', train=True, download=True, transform=f)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Load the data for validation
validation_set = datasets.MNIST(root='./../data', train=False, download=True, transform=f)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=False)
# shuffle=False because validation set is always processed in the same order, which ensures that the evaluation results are consistent across different runs.


class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel(MNIST images are grayscale), 32 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 input channels corresponding to previous output, 64 output channels, 3x3 kernel
        self.pool = nn.MaxPool2d(2, 2)  # Classical pooling layer with 2x2 kernel and stride 2
        self.fc1 = nn.Linear(64*7*7, 128)  # 64 channels/features, 7x7 image size after pooling, 128 output features
        self.fc2 = nn.Linear(128, 10) # 128 input features, 10 output features corresponding to the 10 digits


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply the first convolutional layer, followed by ReLU and pooling
        x = self.pool(F.relu(self.conv2(x)))  # Apply the second convolutional layer, followed by ReLU and pooling
        x = x.view(-1, 64*7*7)  # Flatten the output of the convolutional layers
        x = F.relu(self.fc1(x))  # Apply the first fully connected layer
        x = self.fc2(x)  # Apply the second fully connected layer
        return x


# Create an instance of the CNN class
model = MNIST_CNN()

loss_function = nn.CrossEntropyLoss()  # Cross-entropy is standard loss function for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Actually, I do not understand how Adam works(yet), but it has a fancy name, so let's use it

# Training loop
epochs = 15

for epoch in range(epochs):
    total_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()  # Zero the gradients, since they are accumulated
        output = model(images)
        current_loss = loss_function(output, labels)
        current_loss.backward()
        optimizer.step()
        total_loss += current_loss.item()

    print(f"Loss in epoch {epoch + 1}: {total_loss}")


correct = 0
total = 0
with torch.no_grad():
    for images, labels in validation_loader:
        output = model(images)
        _, predicted = torch.max(output, 1)  # Get the index of the maximum value
        total += labels.size(0)  # Count the number of labels
        correct += (predicted == labels).sum().item()  # Count the number of correct predictions

print(f'Accuracy: {100 * correct / total}%')


# Save the model
torch.save(model.state_dict(), 'mnist_cnn.pth')

# Load the model:
# model.load_state_dict(torch.load('mnist_cnn.pth'))