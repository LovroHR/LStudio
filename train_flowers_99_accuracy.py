import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# FLOWERS DATASET - HIGH ACCURACY VERSION
# Model: EfficientNet_B3
# Cilj: maksimalna točnost (~99% ako dataset dopušta)
# ============================================================

DATA_DIR = "flowers"

BATCH_SIZE = 8
NUM_EPOCHS = 40
LEARNING_RATE = 0.00003
WEIGHT_DECAY = 0.0005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Koristi se:", device)


# ============================================================
# TRANSFORMACIJE
# ============================================================

train_transforms = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomResizedCrop(300),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_transforms = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

base_dataset = datasets.ImageFolder(DATA_DIR)

class_names = base_dataset.classes
num_classes = len(class_names)

print("Klase:", class_names)
print("Broj klasa:", num_classes)
print("Ukupno slika:", len(base_dataset))

train_size = int(0.70 * len(base_dataset))
val_size = int(0.15 * len(base_dataset))
test_size = len(base_dataset) - train_size - val_size

generator = torch.Generator().manual_seed(42)

train_subset, val_subset, test_subset = random_split(
    base_dataset,
    [train_size, val_size, test_size],
    generator=generator
)

train_dataset_full = datasets.ImageFolder(
    DATA_DIR,
    transform=train_transforms
)

test_dataset_full = datasets.ImageFolder(
    DATA_DIR,
    transform=test_transforms
)

train_dataset = Subset(train_dataset_full, train_subset.indices)
val_dataset = Subset(test_dataset_full, val_subset.indices)
test_dataset = Subset(test_dataset_full, test_subset.indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL - EFFICIENTNET_B3
# ============================================================

model = models.efficientnet_b3(
    weights=models.EfficientNet_B3_Weights.DEFAULT
)

# Zamrzni početne slojeve
for param in model.parameters():
    param.requires_grad = False

# Fine tuning zadnjih feature blokova
for param in model.features[-3:].parameters():
    param.requires_grad = True

num_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_features, num_classes)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS
)

writer = SummaryWriter("runs/flowers_99_accuracy")

best_val_acc = 0.0
best_model_weights = copy.deepcopy(model.state_dict())


# ============================================================
# TRENIRANJE
# ============================================================

for epoch in range(NUM_EPOCHS):

    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 50)

    model.train()

    train_loss = 0.0
    train_correct = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * images.size(0)

        train_correct += torch.sum(preds == labels)

    scheduler.step()

    train_loss = train_loss / len(train_dataset)

    train_acc = train_correct.double() / len(train_dataset)

    # VALIDACIJA

    model.eval()

    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            val_loss += loss.item() * images.size(0)

            val_correct += torch.sum(preds == labels)

    val_loss = val_loss / len(val_dataset)

    val_acc = val_correct.double() / len(val_dataset)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/validation", val_loss, epoch)

    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/validation", val_acc, epoch)

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        best_model_weights = copy.deepcopy(model.state_dict())

        torch.save(
            best_model_weights,
            "best_flowers_99_model.pth"
        )

        print("Spremljen najbolji model.")

writer.close()

print("\nNajbolji validation accuracy:",
      round(best_val_acc.item() * 100, 2), "%")


# ============================================================
# TESTIRANJE TESTNIM SETOM PODATAKA
# ============================================================

print("\nTestiranje mreže testnim skupom podataka...")

model.load_state_dict(
    torch.load(
        "best_flowers_99_model.pth",
        map_location=device
    )
)

model.eval()

test_correct = 0

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        test_correct += torch.sum(preds == labels)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = test_correct.double() / len(test_dataset)

print("\n========================================")
print("REZULTATI TESTIRANJA")
print("Test Accuracy:",
      round(test_acc.item() * 100, 2), "%")
print("========================================")

print("\nClassification report:")
print(classification_report(
    all_labels,
    all_preds,
    target_names=class_names
))

print("\nMatrica zabune:")
print(confusion_matrix(
    all_labels,
    all_preds
))

print("\nModel spremljen kao:")
print("best_flowers_99_model.pth")

print("\nTensorBoard:")
print("python -m tensorboard.main --logdir=runs")
