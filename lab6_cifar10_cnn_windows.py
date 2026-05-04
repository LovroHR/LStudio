"""
6. laboratorijska vježba - CIFAR-10 CNN koji radi na Windowsu

Pokretanje:
python lab6_cifar10_cnn_windows.py

TensorBoard:
tensorboard --logdir runs

Napomena:
Ova verzija ima if __name__ == "__main__" i num_workers=0
kako bi radila bez multiprocessing greške na Windowsu.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


DATA_DIR = "./data"
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
SEED = 123
BEST_MODEL_PATH = "najbolji_cifar10_model.pth"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),

            nn.Dropout2d(dropout)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))


class MojCIFARCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 64, dropout=0.05),
            ResidualBlock(64, dropout=0.05),
            nn.MaxPool2d(2),

            ConvBlock(64, 128, dropout=0.10),
            ResidualBlock(128, dropout=0.10),
            nn.MaxPool2d(2),

            ConvBlock(128, 256, dropout=0.15),
            ResidualBlock(256, dropout=0.15),
            nn.MaxPool2d(2),

            ConvBlock(256, 512, dropout=0.20),
            ResidualBlock(512, dropout=0.20),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.45),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx + 1}/{len(loader)} | Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Koristi se uređaj:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        ),
    ])

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    classes = train_dataset.classes
    print("Klase:", classes)
    print("Broj klasa:", len(classes))

    model = MojCIFARCNN(num_classes=10).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    writer = SummaryWriter("runs/lab6_cifar10_vlastiti_cnn_windows")

    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    writer.add_graph(model, dummy_input)

    best_test_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test acc: {test_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)
        writer.add_scalar("Learning_rate", current_lr, epoch)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_test_acc": best_test_acc,
                "classes": classes
            }, BEST_MODEL_PATH)

            print(f"Spremam novi najbolji model: {best_test_acc:.2f}%")

    writer.close()

    print("\nTreniranje završeno.")
    print(f"Najbolja test accuracy: {best_test_acc:.2f}%")
    print(f"Najbolji model spremljen kao: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
