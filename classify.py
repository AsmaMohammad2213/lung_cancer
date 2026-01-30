import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import timm
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
DATA_DIR = r"C:\Users\ASMA\Desktop\lung cancer final\segmented_roi"
MODEL_DIR = "models"
EPOCHS = 15
BATCH_SIZE = 16
LR = 1e-4
TRAIN_SPLIT = 0.8

os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# TRANSFORMS
# ===============================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ===============================
# DATASET
# ===============================
dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
num_classes = len(dataset.classes)

train_size = int(TRAIN_SPLIT * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])
test_ds.dataset.transform = test_transform

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ===============================
# TRAIN FUNCTION
# ===============================
def train_model(model, name):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    best_acc = 0
    best_path = os.path.join(MODEL_DIR, f"{name}.pth")

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"{name} | Epoch {epoch+1}/{EPOCHS} | Train={train_acc:.3f} | Val={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print(f"✅ {name} Best Val Accuracy: {best_acc:.4f}")
    return best_acc

# ===============================
# MODELS
# ===============================
results = {}

# 1️⃣ RS-CapsNet
class RS_CapsNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

print("\n🔹 Training RS-CapsNet")
results["RS-CapsNet"] = train_model(RS_CapsNet(num_classes), "RS_CapsNet")

# 2️⃣ Xception
print("\n🔹 Training Xception")
xception = timm.create_model("xception", pretrained=True, num_classes=num_classes)
for p in xception.parameters():
    p.requires_grad = False
for p in xception.get_classifier().parameters():
    p.requires_grad = True

results["Xception"] = train_model(xception, "Xception")

# 3️⃣ DenseNet-201
print("\n🔹 Training DenseNet-201")
densenet = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
for p in densenet.parameters():
    p.requires_grad = False
densenet.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(densenet.classifier.in_features, num_classes)
)

results["DenseNet-201"] = train_model(densenet, "DenseNet201")

# 4️⃣ EfficientNetV2-B3 (CORRECT NAME)
print("\n🔹 Training EfficientNetV2-B3")
effnet = timm.create_model("tf_efficientnetv2_b3", pretrained=True, num_classes=num_classes)
for p in effnet.parameters():
    p.requires_grad = False
for p in effnet.get_classifier().parameters():
    p.requires_grad = True

results["EfficientNetV2-B3"] = train_model(effnet, "EfficientNetV2B3")

# ===============================
# COMPARATIVE ANALYSIS
# ===============================
plt.figure(figsize=(8,5))
plt.bar(results.keys(), results.values())
plt.ylabel("Validation Accuracy")
plt.title("Model Comparison on Lung Cancer Dataset")
for i, v in enumerate(results.values()):
    plt.text(i, v+0.01, f"{v:.3f}", ha="center")
plt.show()

best_model = max(results, key=results.get)
print("\n🏆 FINAL BEST MODEL:", best_model)
