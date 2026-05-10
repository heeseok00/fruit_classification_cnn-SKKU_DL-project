# ============================================================
# v1 Changes from baseline_code_v0.py:
#   - n_epochs = 1 (assignment constraint)
#   - Data path updated to Data/fruit_data_raw for local execution
#   - Added Normalize (ImageNet mean/std) to both train and test transforms
#   - Added data augmentation to train_tfm
#     (RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter)
#   - Added OneCycleLR scheduler (called per batch for 1-epoch super-convergence)
#   - Fixed optimizer.zero_grad() placement (before forward pass)
#   - Submission file: submission_v1_augment_normalize_onecyclelr.csv
# ============================================================


# ============================================================
# Imports
# ============================================================
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
from tqdm.auto import tqdm


# ============================================================
# Reproducibility
# ============================================================
myseed = 6666
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
# ImageNet mean/std for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Validation / Test: resize + normalize only (no augmentation)
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Train: augmentation + normalize
train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
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
            self.samples = sorted(self.root.glob("*.jpg"))
            self.labels = None
        else:
            assert class_to_idx is not None, "class_to_idx is required for train/valid"
            self.class_to_idx = class_to_idx
            samples, labels = [], []
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
            print(f"Loaded {len(self.samples)} samples from {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]
        im = Image.open(fname).convert("RGB")
        im = self.transform(im)
        if self.is_test:
            return im, -1, fname.stem
        return im, self.labels[idx]


# ============================================================
# Model (unchanged from v0)
# ============================================================
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        # input dimension [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),   # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),        # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),        # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),        # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),        # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),        # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ============================================================
# Dataset Path & Class Mapping
# ============================================================
def resolve_dataset_dir():
    # Kaggle environment
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for child in kaggle_input.iterdir():
            if child.is_dir() and (child / "classes.txt").is_file():
                return child
    # Local environment
    local_path = Path("./Data/fruit_data_raw")
    if local_path.is_dir():
        return local_path
    return Path("./data")


# ============================================================
# Main (required on Windows for num_workers > 0)
# ============================================================
if __name__ == "__main__":

    _dataset_dir = resolve_dataset_dir()
    print(f"Dataset directory: {_dataset_dir}")

    with open(_dataset_dir / "classes.txt", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")

    # ============================================================
    # Data Loaders
    # ============================================================
    batch_size = 64

    train_set = FruitDataset(_dataset_dir / "train", tfm=train_tfm, class_to_idx=class_to_idx)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    valid_set = FruitDataset(_dataset_dir / "valid", tfm=test_tfm, class_to_idx=class_to_idx)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ============================================================
    # Training Setup
    # ============================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    n_epochs = 1  # assignment constraint: 1 epoch only

    model = Classifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # OneCycleLR: ramps LR up then down over all batches in 1 epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs,
    )

    best_acc = 0
    _exp_name = "fruit_classification_v1"

    # ============================================================
    # Training & Validation Loop
    # ============================================================
    for epoch in range(n_epochs):

        # ---------- Training ----------
        model.train()
        train_loss, train_accs = [], []

        for batch in tqdm(train_loader):
            imgs, labels = batch

            optimizer.zero_grad()
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            scheduler.step()  # update LR after every batch

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc  = sum(train_accs) / len(train_accs)
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        model.eval()
        valid_loss, valid_accs = [], []

        for batch in tqdm(valid_loader):
            imgs, labels = batch
            with torch.no_grad():
                logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            acc  = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc  = sum(valid_accs) / len(valid_accs)
        tag = " -> best" if valid_acc > best_acc else ""
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}{tag}")

        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch + 1}, saving model")
            torch.save(model.state_dict(), f"{_exp_name}_best.ckpt")
            best_acc = valid_acc

    # ============================================================
    # Testing
    # ============================================================
    test_set    = FruitDataset(_dataset_dir / "test", tfm=test_tfm, is_test=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model_best = Classifier(num_classes).to(device)
    model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
    model_best.eval()

    prediction, file_ids = [], []
    with torch.no_grad():
        for data, _, file_id in test_loader:
            test_pred  = model_best(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            prediction += test_label.tolist()
            file_ids   += list(file_id)

    # ============================================================
    # Submission
    # ============================================================
    df = pd.DataFrame({"ID": file_ids, "Category": prediction})
    df.to_csv("submission_v1_augment_normalize_onecyclelr.csv", index=False)
    print(f"Wrote submission_v1_augment_normalize_onecyclelr.csv with {len(df)} rows")
