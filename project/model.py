import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import pandas as pd
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.metrics import precision_recall_fscore_support


transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])


dataset = datasets.ImageFolder(
    r"C:/Users/Vedika/PROJECTS/asl alphabet/asl_alphabet_train/asl_alphabet_train", transform= transform
)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size = 32, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 32, shuffle = False)



class Cnn(nn.Module):

    def __init__(self, input_channels, num_classes=29):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
  
  

model = Cnn(input_channels=3, num_classes=29)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)




num_epochs = 10

for epoch in range(num_epochs):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for features, labels in train_loader:

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    scheduler.step(epoch_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Loss: {epoch_loss:.4f}")
    print(f"Accuracy: {epoch_accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("-"*50)



model.eval()

running_loss = 0
correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss = running_loss / len(test_loader)
test_accuracy = 100 * correct / total

precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels,
    all_preds,
    average="macro",
    zero_division=0
)

print("\nTest Results")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_accuracy:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


model_save_path = r"C:/Users/Vedika/PROJECTS/sign_model.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "class_to_idx": dataset.class_to_idx,
    "classes": dataset.classes
}, model_save_path)

torch.save(model.state_dict(), "sign_model.pth")

print(f"Model successfully saved to: {model_save_path}")
