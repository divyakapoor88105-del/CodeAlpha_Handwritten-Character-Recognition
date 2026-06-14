import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train_model import EMNIST_CNN, NUM_CLASSES, label_to_char

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EXTRA_EPOCHS = 5
LR = 0.0005  # lower LR for fine-tuning

# Augmented training transform: small rotation + translation + scale jitter
train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Loading EMNIST byclass dataset...")
train_dataset = datasets.EMNIST(root="./data", split="byclass", train=True,
                                 download=True, transform=train_transform)
test_dataset = datasets.EMNIST(root="./data", split="byclass", train=False,
                                download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

model = EMNIST_CNN(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("emnist_cnn.pth", map_location=DEVICE))
print("Loaded existing emnist_cnn.pth")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


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
    print(f"Epoch {epoch+1}/{EXTRA_EPOCHS} - Train Loss: {running_loss/total:.4f} - Train Acc: {acc:.2f}%")
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
    print("Evaluating current model before fine-tuning...")
    best_acc = evaluate()
    print(f"Starting accuracy: {best_acc:.2f}%")

    for epoch in range(EXTRA_EPOCHS):
        train_one_epoch(epoch)
        test_acc = evaluate()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "emnist_cnn.pth")
            print(f"  -> Saved improved model ({best_acc:.2f}%)")
        else:
            print(f"  -> No improvement (best stays {best_acc:.2f}%)")

    print(f"\nFine-tuning complete. Best test accuracy: {best_acc:.2f}%")