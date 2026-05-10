# v7 — stage당 ResBlock 2개 → 3개 (더 깊은 모델, v2 대비 변경)
# Submission: submission_v7_deeper_resblock.csv

import ctypes, random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(myseed); np.random.seed(myseed); torch.manual_seed(myseed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(myseed)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)), transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
train_tfm = transforms.Compose([
    transforms.Resize((128, 128)), transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

class FruitDataset(Dataset):
    def __init__(self, root, tfm=test_tfm, class_to_idx=None, is_test=False):
        super().__init__()
        self.root = Path(root); self.transform = tfm; self.is_test = is_test
        if is_test:
            self.samples = sorted(self.root.glob("*.jpg")); self.labels = None
        else:
            assert class_to_idx is not None
            self.class_to_idx = class_to_idx
            samples, labels = [], []
            for class_dir in sorted(self.root.iterdir()):
                if not class_dir.is_dir(): continue
                label = class_to_idx[class_dir.name]
                for img in sorted(class_dir.glob("*.jpg")):
                    samples.append(img); labels.append(label)
            self.samples = samples; self.labels = labels
        print(f"Loaded {len(self.samples)} samples from {self.root}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        fname = self.samples[idx]
        im = self.transform(Image.open(fname).convert("RGB"))
        return (im, -1, fname.stem) if self.is_test else (im, self.labels[idx])

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if stride != 1 or in_channels != out_channels else nn.Identity()
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + self.shortcut(x))

# ── 변경: stage당 ResBlock 3개 (v2는 2개) ──
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64), ResBlock(64, 128, stride=2))
        self.stage2 = nn.Sequential(ResBlock(128, 128), ResBlock(128, 128), ResBlock(128, 256, stride=2))
        self.stage3 = nn.Sequential(ResBlock(256, 256), ResBlock(256, 256), ResBlock(256, 512, stride=2))
        self.stage4 = nn.Sequential(ResBlock(512, 512), ResBlock(512, 512), ResBlock(512, 512))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.stem(x); x = self.stage1(x); x = self.stage2(x)
        x = self.stage3(x); x = self.stage4(x)
        return self.fc(self.gap(x).view(x.size(0), -1))

def resolve_dataset_dir():
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for child in kaggle_input.iterdir():
            if child.is_dir() and (child / "classes.txt").is_file(): return child
    local = Path("./Data/fruit_data_raw")
    return local if local.is_dir() else Path("./data")

if __name__ == "__main__":
    _dataset_dir = resolve_dataset_dir()
    with open(_dataset_dir / "classes.txt", encoding="utf-8") as f:
        classes = [l.strip() for l in f if l.strip()]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes  = len(classes)

    _tmp = Classifier(num_classes)
    total_params = sum(p.numel() for p in _tmp.parameters())
    print(f"v7 | Deeper ResBlock (3/stage) | Total params: {total_params:,}")
    del _tmp

    batch_size = 64

    train_set    = FruitDataset(_dataset_dir / "train", tfm=train_tfm, class_to_idx=class_to_idx)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    valid_set    = FruitDataset(_dataset_dir / "valid", tfm=test_tfm,  class_to_idx=class_to_idx)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epochs  = 1
    model     = Classifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=n_epochs)
    best_acc  = 0; _exp_name = "fruit_classification_v7"
    print(f"Device={device}")

    history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
    for epoch in range(n_epochs):
        model.train(); train_loss, train_accs = [], []
        for imgs, labels in tqdm(train_loader, desc=f"[Train {epoch+1}/{n_epochs}]"):
            optimizer.zero_grad()
            logits = model(imgs.to(device)); loss = criterion(logits, labels.to(device))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step(); scheduler.step()
            train_loss.append(loss.item()); train_accs.append((logits.argmax(-1)==labels.to(device)).float().mean().item())
        t_loss = sum(train_loss)/len(train_loss); t_acc = sum(train_accs)/len(train_accs)
        print(f"[ Train | {epoch+1:03d}/{n_epochs:03d} ] loss={t_loss:.5f}, acc={t_acc:.5f}")

        model.eval(); valid_loss, valid_accs = [], []
        for imgs, labels in tqdm(valid_loader, desc=f"[Valid {epoch+1}/{n_epochs}]"):
            with torch.no_grad(): logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            valid_loss.append(loss.item()); valid_accs.append((logits.argmax(-1)==labels.to(device)).float().mean().item())
        v_loss = sum(valid_loss)/len(valid_loss); v_acc = sum(valid_accs)/len(valid_accs)
        tag = " -> best" if v_acc > best_acc else ""
        print(f"[ Valid | {epoch+1:03d}/{n_epochs:03d} ] loss={v_loss:.5f}, acc={v_acc:.5f}{tag}")
        history["train_loss"].append(t_loss); history["train_acc"].append(t_acc)
        history["valid_loss"].append(v_loss); history["valid_acc"].append(v_acc)
        if v_acc > best_acc:
            torch.save(model.state_dict(), f"{_exp_name}_best.ckpt"); best_acc = v_acc
    print(f"\nTraining done. Best valid acc: {best_acc:.5f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss", marker="o"); axes[0].plot(history["valid_loss"], label="Valid Loss", marker="o")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history["train_acc"], label="Train Acc", marker="o"); axes[1].plot(history["valid_acc"], label="Valid Acc", marker="o")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True)
    plt.suptitle("v7 — Deeper ResBlock (3/stage)", fontsize=13); plt.tight_layout()
    plt.savefig("v7_training_curve.png", dpi=150); plt.close()

    test_set    = FruitDataset(_dataset_dir / "test", tfm=test_tfm, is_test=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model_best  = Classifier(num_classes).to(device)
    model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt")); model_best.eval()
    prediction, file_ids = [], []
    with torch.no_grad():
        for data, _, file_id in tqdm(test_loader, desc="Testing"):
            test_label = np.argmax(model_best(data.to(device)).cpu().numpy(), axis=1)
            prediction += test_label.tolist(); file_ids += list(file_id)

    submission_dir = Path("./Data/submissions"); submission_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"ID": file_ids, "Category": prediction})
    df.to_csv(submission_dir / "submission_v7_deeper_resblock.csv", index=False)
    print(f"Saved: {submission_dir / 'submission_v7_deeper_resblock.csv'}")
    ctypes.windll.user32.MessageBoxW(0, "v7 Done! Deeper ResBlock 3/stage", "Done", 1)
