# data loading

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Basic transformations
transform = transforms.Compose([
    transforms.Grayscale(),        # In case images have channels
    transforms.Resize((48, 48)),
    transforms.ToTensor(),         # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

# Load train and test datasets
train_dataset = datasets.ImageFolder(root=r"C:\Users\jenit\Downloads\scraping", transform=transform)
test_dataset = datasets.ImageFolder(root=r"C:\Users\jenit\Downloads\scraping", transform=transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(train_dataset.classes)  # ['angry', 'disgusted', ..., 'neutral']




### Building a simple CNN model

import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)

        # Dummy forward pass to compute flattened size
        dummy_input = torch.zeros(1, 1, 48, 48)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        self.flattened_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Automatically handle batch size
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


### Training the model

import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(40):  # Change epoch count as needed
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


## Model evaluation

correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")



import os
from PIL import Image
import matplotlib.pyplot as plt

# Folder containing custom images for prediction
custom_image_dir = r"C:\Users\jenit\Downloads\testings"  # <- change this to your folder

# Collect all image paths in the folder
image_paths = []
for file in os.listdir(custom_image_dir):
    if file.endswith(('.jpg', '.jpeg', '.png')):
        image_paths.append(os.path.join(custom_image_dir, file))

# Class labels
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Prediction loop
model.eval()
for image_path in image_paths:
    image = Image.open(image_path).convert('L')  # convert to grayscale
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted_emotion = classes[predicted.item()]

    # Show each image with prediction
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {predicted_emotion}")
    plt.axis('off')
    plt.show()
