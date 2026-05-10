# ============================================================
# Simple CNN baseline for the Fruits-360 (re-split, anonymized) Kaggle competition.
# Adapted from a food-11 baseline. Key changes:
#   - labels come from class folder names (mapped via classes.txt)
#   - 257 output classes
#   - test set is a flat folder of anonymized image IDs
#   - data path auto-detects /kaggle/input/, falls back to ./data
# ============================================================


# ============================================================
# Imports
# ============================================================
# Import necessary packages.
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# This is for the progress bar.
from tqdm.auto import tqdm


# ============================================================
# Reproducibility
# ============================================================
myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(myseed)
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


# ============================================================
# Data Transforms
# ============================================================
# Normally, we don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])


# ============================================================
# Dataset
# ============================================================
class FruitDataset(Dataset):
    """Train/valid: walks class subfolders. Test: flat folder of anonymized jpgs."""

    def __init__(self, root, tfm=test_tfm, class_to_idx=None, is_test=False):
        super().__init__()
        self.root = Path(root)
        self.transform = tfm
        self.is_test = is_test

        if is_test:
            # Test set: flat folder of anonymized jpgs (no class subfolders, no labels).
            self.samples = sorted(self.root.glob("*.jpg"))
            self.labels = None
        else:
            # Train/valid: walk each class subfolder and assign labels via class_to_idx.
            assert class_to_idx is not None, "class_to_idx is required for train/valid"
            self.class_to_idx = class_to_idx
            samples = []
            labels = []
            for class_dir in sorted(self.root.iterdir()):
                if not class_dir.is_dir():
                    continue
                if class_dir.name not in class_to_idx:
                    raise ValueError(f"Unknown class folder: {class_dir.name}")
                label = class_to_idx[class_dir.name]
                for img in sorted(class_dir.glob("*.jpg")):
                    samples.append(img)
                    labels.append(label)
            self.samples = samples
            self.labels = labels

        if self.samples:
            print(f"One {self.root} sample: {self.samples[0]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]
        im = Image.open(fname).convert("RGB")
        im = self.transform(im)
        if self.is_test:
            # Return file_id (filename stem) only for test set.
            return im, -1, fname.stem
        else:
            return im, self.labels[idx]


# ============================================================
# Model
# ============================================================
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input dimension [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ============================================================
# Dataset Path & Class Mapping
# ============================================================
def resolve_dataset_dir():
    # Auto-detect Kaggle input directory; fall back to ./data when run locally.
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for child in kaggle_input.iterdir():
            if child.is_dir() and (child / "classes.txt").is_file():
                return child
    return Path("./data")


_dataset_dir = resolve_dataset_dir()
print(f"Dataset directory: {_dataset_dir}")

# Load class names from classes.txt and build class_to_idx mapping.
with open(_dataset_dir / "classes.txt", encoding="utf-8") as f:
    classes = [line.strip() for line in f if line.strip()]
class_to_idx = {c: i for i, c in enumerate(classes)}
num_classes = len(classes)
print(f"Number of classes: {num_classes}")


# ============================================================
# Data Loaders
# ============================================================
batch_size = 64
# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = FruitDataset(_dataset_dir / "train", tfm=train_tfm, class_to_idx=class_to_idx)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = FruitDataset(_dataset_dir / "valid", tfm=test_tfm, class_to_idx=class_to_idx)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


# ============================================================
# Training Setup
# ============================================================
# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# The number of training epochs and patience.
n_epochs = 10
patience = 300  # If no improvement in 'patience' epochs, early stop

# Initialize a model, and put it on the device specified.
model = Classifier(num_classes).to(device)

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0
_exp_name = "fruit_classification_baseline"


# ============================================================
# Training & Validation Loop
# ============================================================
for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        # imgs = imgs.half()
        # print(imgs.shape,labels.shape)

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        # imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        # break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    tag = " -> best" if valid_acc > best_acc else ""
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}{tag}")

    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt")  # only save best to prevent output memory exceed error
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvement for {patience} consecutive epochs, early stopping")
            break


# ============================================================
# Testing
# ============================================================
test_set = FruitDataset(_dataset_dir / "test", tfm=test_tfm, is_test=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

model_best = Classifier(num_classes).to(device)
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
model_best.eval()
prediction = []
file_ids = []  # To store file IDs for each image

with torch.no_grad():
    for data, _, file_id in test_loader:
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.tolist()
        file_ids += list(file_id)  # Append original IDs


# ============================================================
# Submission
# ============================================================
# create test csv
df = pd.DataFrame({"ID": file_ids, "Category": prediction})
df.to_csv("submission.csv", index=False)
print(f"Wrote submission.csv with {len(df)} rows")
