import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# 8. LABORATORIJSKA VJEŽBA
# CIFAR-10 + vlastita ResNet arhitektura
# Windows verzija, manje epoha
# ============================================================

BATCH_SIZE = 128
NUM_EPOCHS = 35
LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Koristi se:", device)


train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


train_full_augmented = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transforms
)

train_full_clean = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=test_transforms
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transforms
)

train_size = int(0.9 * len(train_full_augmented))

indices = torch.randperm(
    len(train_full_augmented),
    generator=torch.Generator().manual_seed(42)
).tolist()

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(train_full_augmented, train_indices)
val_dataset = Subset(train_full_clean, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))
print("Test size:", len(test_dataset))


# ============================================================
# VLASTITA RESNET ARHITEKTURA
# ============================================================

class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class CustomResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []

        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.dropout(out)
        out = self.fc(out)

        return out


def custom_resnet18(num_classes=10):
    return CustomResNet(
        ResidualBlock,
        [2, 2, 2, 2],
        num_classes=num_classes
    )


model = custom_resnet18(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS
)

writer = SummaryWriter("runs/lab8_cifar10_custom_resnet_fast")

best_val_acc = 0.0


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)

        running_loss += loss.item() * images.size(0)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_predictions, all_labels


for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 50)

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    val_loss, val_acc, _, _ = evaluate(
        model,
        val_loader,
        criterion
    )

    scheduler.step()

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc * 100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/validation", val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/validation", val_acc, epoch)
    writer.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], epoch)

    if val_acc > best_val_acc:
        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "best_custom_resnet_cifar10_fast.pth"
        )

        print("Spremljen novi najbolji model.")


writer.close()

print("\nNajbolji validation accuracy:",
      round(best_val_acc * 100, 2), "%")


print("\nUčitavam najbolji model i testiram mrežu na CIFAR-10 testnom setu...")

model.load_state_dict(
    torch.load(
        "best_custom_resnet_cifar10_fast.pth",
        map_location=device
    )
)

test_loss, test_acc, test_predictions, test_labels = evaluate(
    model,
    test_loader,
    criterion
)

print("\n==============================================")
print("REZULTATI TESTIRANJA NA TESTNOM SETU")
print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_acc * 100, 2), "%")
print("==============================================")

print("\nClassification report:")
print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=class_names
    )
)

print("\nMatrica zabune:")
print(
    confusion_matrix(
        test_labels,
        test_predictions
    )
)

print("\nModel je spremljen kao:")
print("best_custom_resnet_cifar10_fast.pth")

print("\nTensorBoard pokreni naredbom:")
print("python -m tensorboard.main --logdir=runs")
