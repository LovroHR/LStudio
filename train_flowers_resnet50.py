import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# 7. LABORATORIJSKA VJEŽBA - TRANSFER LEARNING
# Dataset: flowers
# Model: ResNet50
# TensorBoard + testiranje testnim skupom podataka
# ============================================================

DATA_DIR = "flowers"
BATCH_SIZE = 16
NUM_EPOCHS = 25
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Koristi se uređaj:", device)


# ============================================================
# TRANSFORMACIJE
# ============================================================

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# Očekivana struktura:
#
# flowers/
#   daisy/
#   dandelion/
#   rose/
#   sunflower/
#   tulip/
# ============================================================

base_dataset = datasets.ImageFolder(DATA_DIR)

class_names = base_dataset.classes
num_classes = len(class_names)

print("Klase:", class_names)
print("Broj klasa:", num_classes)
print("Ukupan broj slika:", len(base_dataset))

train_size = int(0.70 * len(base_dataset))
val_size = int(0.15 * len(base_dataset))
test_size = len(base_dataset) - train_size - val_size

generator = torch.Generator().manual_seed(42)

train_subset, val_subset, test_subset = random_split(
    base_dataset,
    [train_size, val_size, test_size],
    generator=generator
)

# Posebni dataset objekti zbog različitih transformacija za train i test/val
train_dataset_full = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
test_dataset_full = datasets.ImageFolder(DATA_DIR, transform=test_transforms)

train_dataset = Subset(train_dataset_full, train_subset.indices)
val_dataset = Subset(test_dataset_full, val_subset.indices)
test_dataset = Subset(test_dataset_full, test_subset.indices)

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

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


# ============================================================
# MODEL - RESNET50 TRANSFER LEARNING
# ============================================================

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Zamrzavanje većine slojeva
for param in model.parameters():
    param.requires_grad = False

# Fine-tuning zadnjeg ResNet bloka za bolju točnost
for param in model.layer4.parameters():
    param.requires_grad = True

# Zamjena završnog sloja
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, num_classes)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)

writer = SummaryWriter("runs/flowers_resnet50")

best_val_acc = 0.0
best_model_weights = copy.deepcopy(model.state_dict())


# ============================================================
# TRENIRANJE I VALIDACIJA
# ============================================================

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 40)

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

    train_loss = train_loss / len(train_dataset)
    train_acc = train_correct.double() / len(train_dataset)

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

    scheduler.step(val_acc)

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
        torch.save(best_model_weights, "best_flowers_resnet50.pth")
        print("Spremljen novi najbolji model.")

writer.close()

print("\nNajbolji validation accuracy:", round(best_val_acc.item() * 100, 2), "%")


# ============================================================
# TESTIRANJE MREŽE TESTNIM SETOM PODATAKA
# ============================================================

print("\nUčitavam najbolji model i testiram mrežu testnim skupom podataka...")

model.load_state_dict(torch.load("best_flowers_resnet50.pth", map_location=device))
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

print("\n==========================================")
print("REZULTATI TESTIRANJA NA TESTNOM SETU")
print("Test accuracy:", round(test_acc.item() * 100, 2), "%")
print("==========================================")

print("\nClassification report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

print("Matrica zabune:")
print(confusion_matrix(all_labels, all_preds))

print("\nModel spremljen kao: best_flowers_resnet50.pth")
print("TensorBoard pokreni naredbom:")
print("python -m tensorboard.main --logdir=runs")
