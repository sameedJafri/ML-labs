"""
Train a DeepLabV3+ (ResNet-50 backbone) segmentation model on the
SAM-generated aerial building dataset produced by prepare_dataset.py.

Usage:
    python train.py

Outputs:
    weights/segmentation_model.pth  – best checkpoint (highest val IoU)
    outputs/training_curves.png
    outputs/sample_predictions.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)
from dotenv import load_dotenv

load_dotenv()

# ── Config from environment ──────────────────────────────────────────────────
DEVICE     = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR   = Path(os.getenv("DATASET_DIR", "data"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", "weights/segmentation_model.pth"))
EPOCHS     = int(os.getenv("EPOCHS", "10"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
LR         = float(os.getenv("LR", "1e-4"))
IMG_SIZE   = 256
OUTPUT_DIR = Path("outputs")


# ── Dataset ──────────────────────────────────────────────────────────────────

class BuildingDataset(Dataset):
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD  = [0.229, 0.224, 0.225]

    def __init__(self, manifest_path: Path, split: str):
        with open(manifest_path) as f:
            data = json.load(f)
        self.samples = data[split]

        self.img_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(self.IMG_MEAN, self.IMG_STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s    = self.samples[idx]
        img  = Image.open(s["image"]).convert("RGB")
        mask = Image.open(s["mask"]).convert("L")

        img_t  = self.img_tf(img)
        mask_t = transforms.functional.resize(
            mask, (IMG_SIZE, IMG_SIZE),
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        mask_t = torch.from_numpy(np.array(mask_t) > 127).long()
        return img_t, mask_t


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred   = pred.bool()
    target = target.bool()
    intersection = (pred & target).float().sum().item()
    union        = (pred | target).float().sum().item()
    return intersection / (union + 1e-8)


def compute_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred   = pred.bool()
    target = target.bool()
    intersection = (pred & target).float().sum().item()
    return 2.0 * intersection / (pred.float().sum().item() + target.float().sum().item() + 1e-8)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model() -> nn.Module:
    """DeepLabV3+ pretrained on COCO, classifier replaced for binary segmentation."""
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    # 2 output classes: 0 = background, 1 = building
    model.classifier[4]     = nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    return model


# ── Train / eval loops ────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        output = model(imgs)
        # DeepLabV3 returns {"out": ..., "aux": ...}
        loss = criterion(output["out"], masks)
        if "aux" in output:
            loss += 0.4 * criterion(output["aux"], masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[float, float]:
    model.eval()
    iou_scores, dice_scores = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)["out"].argmax(dim=1)
        for pred, mask in zip(preds, masks):
            iou_scores.append(compute_iou(pred, mask))
            dice_scores.append(compute_dice(pred, mask))
    return float(np.mean(iou_scores)), float(np.mean(dice_scores))


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_curves(history: dict, out_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(epochs, history["val_iou"], label="Val IoU", color="green")
    axes[1].set_title("Validation IoU"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    axes[2].plot(epochs, history["val_dice"], label="Val Dice", color="orange")
    axes[2].set_title("Validation Dice"); axes[2].set_xlabel("Epoch"); axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close(fig)
    print(f"  Saved training_curves.png")


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


@torch.no_grad()
def plot_predictions(
    model: nn.Module,
    loader: DataLoader,
    out_dir: Path,
    n: int = 4,
) -> None:
    model.eval()
    imgs, masks = next(iter(loader))
    imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
    preds = model(imgs)["out"].argmax(dim=1)

    fig, axes = plt.subplots(n, 3, figsize=(9, n * 3))
    titles = ["Aerial Image", "SAM Mask (GT)", "Predicted Mask"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=12)

    for row in range(min(n, len(imgs))):
        axes[row, 0].imshow(denormalize(imgs[row]))
        axes[row, 1].imshow(masks[row].cpu(), cmap="gray")
        axes[row, 2].imshow(preds[row].cpu(), cmap="gray")
        for col in range(3):
            axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "sample_predictions.png", dpi=120)
    plt.close(fig)
    print(f"  Saved sample_predictions.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def train() -> None:
    manifest = DATA_DIR / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest}. Run prepare_dataset.py first."
        )

    train_ds = BuildingDataset(manifest, "train")
    val_ds   = BuildingDataset(manifest, "val")
    test_ds  = BuildingDataset(manifest, "test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Dataset  — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    model     = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history  = {"train_loss": [], "val_iou": [], "val_dice": []}
    best_iou = 0.0

    print(f"\nTraining DeepLabV3+ on {DEVICE} for {EPOCHS} epochs...")
    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val IoU':>8}  {'Val Dice':>9}")
    print("-" * 45)

    for epoch in range(1, EPOCHS + 1):
        train_loss         = train_one_epoch(model, train_loader, optimizer, criterion)
        val_iou, val_dice  = evaluate(model, val_loader)

        history["train_loss"].append(train_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)

        marker = " *" if val_iou > best_iou else ""
        print(f"{epoch:>6}  {train_loss:>11.4f}  {val_iou:>8.4f}  {val_dice:>9.4f}{marker}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), MODEL_PATH)

    # ── Final test evaluation ────────────────────────────────────────────────
    print(f"\nLoading best model (IoU={best_iou:.4f}) for test evaluation...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    test_iou, test_dice = evaluate(model, test_loader)
    print(f"Test Set  →  IoU: {test_iou:.4f}   Dice: {test_dice:.4f}")

    # ── Visualisations ───────────────────────────────────────────────────────
    print(f"\nSaving visualisations to {OUTPUT_DIR}/...")
    plot_curves(history, OUTPUT_DIR)
    plot_predictions(model, test_loader, OUTPUT_DIR)

    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
