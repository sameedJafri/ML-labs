"""
Dataset preparation using Meta's Segment Anything Model (SAM).

Week 7 approach: use SamAutomaticMaskGenerator on aerial images, then
filter masks by area and stability to isolate building-like segments.

Usage:
    # 1. Download SAM checkpoint:
    #    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    # 2. Install SAM:
    #    pip install git+https://github.com/facebookresearch/segment-anything.git
    # 3. Run:
    #    python prepare_dataset.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from dotenv import load_dotenv

load_dotenv()

# ── Config from environment ──────────────────────────────────────────────────
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", "sam_vit_h_4b8939.pth")
DEVICE         = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR    = Path(os.getenv("DATASET_DIR", "data"))
HF_TOKEN       = os.getenv("HUGGINGFACE_TOKEN")        # optional, for private HF datasets

# SAM mask filtering thresholds for buildings
MIN_AREA       = int(os.getenv("SAM_MIN_AREA", "300"))
MAX_AREA       = int(os.getenv("SAM_MAX_AREA", "40000"))
MIN_STABILITY  = float(os.getenv("SAM_MIN_STABILITY", "0.85"))

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
# remainder → test


def download_aerial_images() -> list[tuple[Image.Image, str]]:
    """
    Download aerial building images from HuggingFace.
    Returns a list of (PIL Image, filename) tuples.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print("Downloading keremberke/satellite-building-segmentation from HuggingFace...")
    kwargs = {"trust_remote_code": True}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN

    ds = load_dataset("keremberke/satellite-building-segmentation", "full", **kwargs)

    samples: list[tuple[Image.Image, str]] = []
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        for idx, item in enumerate(ds[split]):
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            samples.append((img.convert("RGB"), f"{split}_{idx:04d}"))

    print(f"  Downloaded {len(samples)} images total.")
    return samples


def masks_to_binary(masks: list[dict]) -> np.ndarray:
    """
    Merge SAM masks into a single binary building mask.
    Filters by area and stability_score to keep only building-like regions.
    """
    if not masks:
        h, w = 256, 256
        return np.zeros((h, w), dtype=np.uint8)

    h, w = masks[0]["segmentation"].shape
    combined = np.zeros((h, w), dtype=np.uint8)

    for m in masks:
        area      = m["area"]
        stability = m["stability_score"]
        if MIN_AREA < area < MAX_AREA and stability > MIN_STABILITY:
            combined = np.maximum(combined, m["segmentation"].astype(np.uint8))

    return combined


def generate_masks(
    samples: list[tuple[Image.Image, str]],
) -> list[tuple[Image.Image, np.ndarray, str]]:
    """
    Run SAM on each image and return (image, binary_mask, name) triples.
    This mirrors the Week 7 SAM usage:
        sam_model_registry[model_type](checkpoint=sam_checkpoint)
        SamAutomaticMaskGenerator(sam).generate(np.array(image))
    """
    # ── Week 7 SAM setup ────────────────────────────────────────────────────
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        raise ImportError(
            "Install SAM: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

    if not Path(SAM_CHECKPOINT).exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at '{SAM_CHECKPOINT}'.\n"
            "Download it with:\n"
            "  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        )

    print(f"Loading SAM (vit_h) on {DEVICE}...")
    sam_checkpoint = SAM_CHECKPOINT
    model_type     = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,           # fewer points → faster, still covers rooftops
        pred_iou_thresh=0.88,
        stability_score_thresh=MIN_STABILITY,
        min_mask_region_area=MIN_AREA,
    )
    # ────────────────────────────────────────────────────────────────────────

    results = []
    for i, (img, name) in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] Generating masks for {name}...")
        masks = mask_generator.generate(np.array(img))      # Week 7 line
        binary_mask = masks_to_binary(masks)
        results.append((img, binary_mask, name))

    return results


def save_split(
    items: list[tuple[Image.Image, np.ndarray, str]],
    split: str,
) -> list[dict]:
    img_dir  = DATASET_DIR / "images" / split
    mask_dir = DATASET_DIR / "masks"  / split
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for img, mask, name in items:
        img_path  = img_dir  / f"{name}.png"
        mask_path = mask_dir / f"{name}.png"

        img.save(img_path)
        Image.fromarray(mask * 255).save(mask_path)

        manifest.append({"image": str(img_path), "mask": str(mask_path)})

    return manifest


def prepare_dataset() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    samples = download_aerial_images()
    results = generate_masks(samples)

    # ── Train / val / test split ─────────────────────────────────────────────
    n        = len(results)
    n_train  = int(n * TRAIN_RATIO)
    n_val    = int(n * VAL_RATIO)

    splits = {
        "train": results[:n_train],
        "val":   results[n_train : n_train + n_val],
        "test":  results[n_train + n_val :],
    }

    manifest: dict[str, list] = {}
    for split, items in splits.items():
        print(f"\nSaving {split} split ({len(items)} samples)...")
        manifest[split] = save_split(items, split)

    manifest_path = DATASET_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Manifest saved to {manifest_path}")
    print(f"  train: {len(manifest['train'])}  val: {len(manifest['val'])}  test: {len(manifest['test'])}")


if __name__ == "__main__":
    prepare_dataset()
