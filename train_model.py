"""
EMNIST CNN Training Script
Trains a CNN to recognize handwritten digits and letters (A-Z, a-z)
Dataset: EMNIST 'byclass' split (62 classes: 0-9, A-Z, a-z)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.001
NUM_CLASSES = 62  # 10 digits + 26 upper + 26 lower

# EMNIST byclass label mapping (index -> character)
# 0-9 = digits, 10-35 = A-Z, 36-61 = a-z
def label_to_char(label):
    if label < 10:
        return str(label)
    elif label < 36:
        return chr(label - 10 + ord('A'))
    else:
        return chr(label - 36 + ord('a'))

# -------------------------
# Data
# -------------------------
# EMNIST images come transposed (28x28) compared to MNIST orientation.
# We rotate/flip to make them look "normal" (upright) for the user later.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Downloading / loading EMNIST byclass dataset...")
train_dataset = datasets.EMNIST(root="./data", split="byclass", train=True,
                                 download=True, transform=transform)
test_dataset = datasets.EMNIST(root="./data", split="byclass", train=False,
                                download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# -------------------------
# Model
# -------------------------
class EMNIST_CNN(nn.Module):
    def __init__(self, num_classes=62):
        super(EMNIST_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7 -> 3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = EMNIST_CNN(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# -------------------------
# Training Loop
# -------------------------
def train_one_epoch(epoch):
    model.train()
    running_loss, correct, total = 0, 0, 0
    num_batches = len(train_loader)
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{num_batches} - Loss: {loss.item():.4f}")

    acc = 100. * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {running_loss/total:.4f} - Train Acc: {acc:.2f}%")
    return acc

def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    acc = 100. * correct / total
    print(f"Test Acc: {acc:.2f}%")
    return acc

if __name__ == "__main__":
    best_acc = 0
    for epoch in range(EPOCHS):
        train_one_epoch(epoch)
        test_acc = evaluate()
        scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "emnist_cnn.pth")
            print(f"  -> Saved new best model ({best_acc:.2f}%)")

    print(f"\nTraining complete. Best test accuracy: {best_acc:.2f}%")
    print("Model saved to emnist_cnn.pth")
    print("Now run: python export_model.py  (to convert for deployment)")
