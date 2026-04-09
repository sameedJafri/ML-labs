"""
Dataset preparation for aerial building segmentation.

Two modes controlled by the USE_SAM environment variable:

  USE_SAM=false (default)
    Uses the ground-truth polygon annotations that ship with the
    keremberke/satellite-building-segmentation dataset.
    No GPU or large model needed — runs in seconds.

  USE_SAM=true
    Week 7 approach: runs SamAutomaticMaskGenerator on each image,
    then filters masks by area and stability to isolate buildings.
    Requires the SAM checkpoint and ~4 GB RAM (ViT-B) or ~8 GB (ViT-H).
    Download checkpoint:
      ViT-B: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
      ViT-H: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Usage:
    python prepare_dataset.py
"""

import os
import gc
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from dotenv import load_dotenv

load_dotenv()

# ── Config from environment ──────────────────────────────────────────────────
USE_SAM        = os.getenv("USE_SAM", "false").lower() == "true"
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", "sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_b")
DEVICE         = os.getenv("DEVICE", "cpu")
DATASET_DIR    = Path(os.getenv("DATASET_DIR", "data"))
HF_TOKEN       = os.getenv("HUGGINGFACE_TOKEN")   # optional, for private HF datasets
# Cap total images to avoid OOM — 600 gives ~420 train / 90 val / 90 test
MAX_SAMPLES    = int(os.getenv("MAX_SAMPLES", "600"))

# SAM mask filtering thresholds (only used when USE_SAM=true)
MIN_AREA      = int(os.getenv("SAM_MIN_AREA", "300"))
MAX_AREA      = int(os.getenv("SAM_MAX_AREA", "40000"))
MIN_STABILITY = float(os.getenv("SAM_MIN_STABILITY", "0.85"))

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
# remainder → test


# ── Annotation helpers ────────────────────────────────────────────────────────

def annotations_to_mask(item: dict, img_size: tuple[int, int]) -> np.ndarray:
    """
    Convert COCO-style polygon segmentation annotations to a binary mask.
    img_size is (width, height).  Returns uint8 array: 0=background, 1=building.

    Dataset format: segmentation[i] = [[x1,y1,x2,y2,...]] — each object's
    polygons are wrapped in an extra list, so we unwrap one level before parsing.
    """
    w, h = img_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for poly_list in item.get("objects", {}).get("segmentation", []):
        # poly_list is [[x1,y1,x2,y2,...]] — unwrap the outer list
        for seg in poly_list:
            if not seg or len(seg) < 6:
                continue
            coords = [(float(seg[i]), float(seg[i + 1])) for i in range(0, len(seg) - 1, 2)]
            if len(coords) >= 3:
                draw.polygon(coords, fill=1)
    return np.array(mask, dtype=np.uint8)


# ── Mode B: Week 7 SAM helpers ────────────────────────────────────────────────

def masks_to_binary(masks: list[dict]) -> np.ndarray:
    """Merge SAM masks into one binary building mask (Week 7 filter logic)."""
    if not masks:
        return np.zeros((256, 256), dtype=np.uint8)
    h, w = masks[0]["segmentation"].shape
    combined = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        if MIN_AREA < m["area"] < MAX_AREA and m["stability_score"] > MIN_STABILITY:
            combined = np.maximum(combined, m["segmentation"].astype(np.uint8))
    return combined


def load_sam_generator():
    """Load SAM model and return a mask generator (Week 7 setup)."""
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        raise ImportError(
            "Install SAM: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )
    if not Path(SAM_CHECKPOINT).exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at '{SAM_CHECKPOINT}'.\n"
            "Download ViT-B (375 MB):\n"
            "  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        )
    print(f"Loading SAM ({SAM_MODEL_TYPE}) on {DEVICE}...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=MIN_STABILITY,
        min_mask_region_area=MIN_AREA,
    )


# ── Core streaming pipeline ───────────────────────────────────────────────────

def stream_and_save() -> None:
    """
    Stream images one at a time from HuggingFace, generate masks, and write
    directly to disk.  Never holds more than one image in RAM at a time.
    Respects MAX_SAMPLES to keep memory and time manageable.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install 'datasets>=2.18.0,<3.0.0'")

    print("Streaming keremberke/satellite-building-segmentation from HuggingFace...")
    kwargs = {"trust_remote_code": True, "streaming": True}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    ds = load_dataset("keremberke/satellite-building-segmentation", "full", **kwargs)

    sam_gen = load_sam_generator() if USE_SAM else None
    if USE_SAM:
        print("Mode: SAM mask generation (Week 7)")
    else:
        print("Mode: built-in dataset annotations (fast, no model needed)")

    # Per-split caps derived from MAX_SAMPLES and the train/val/test ratios
    n_train = int(MAX_SAMPLES * TRAIN_RATIO)
    n_val   = int(MAX_SAMPLES * VAL_RATIO)
    n_test  = MAX_SAMPLES - n_train - n_val
    per_split_cap = {"train": n_train, "validation": n_val, "test": n_test}

    manifest: dict[str, list] = {"train": [], "val": [], "test": []}

    for hf_split, cap in per_split_cap.items():
        if hf_split not in ds or cap == 0:
            continue
        split_tag = "val" if hf_split == "validation" else hf_split

        img_dir  = DATASET_DIR / "images" / split_tag
        mask_dir = DATASET_DIR / "masks"  / split_tag
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for idx, item in enumerate(ds[hf_split]):
            if idx >= cap:
                break

            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB")

            if USE_SAM:
                masks  = sam_gen.generate(np.array(img))   # Week 7 line
                binary = masks_to_binary(masks)
            else:
                binary = annotations_to_mask(item, img.size)

            name      = f"{split_tag}_{idx:04d}"
            img_path  = img_dir  / f"{name}.png"
            mask_path = mask_dir / f"{name}.png"

            img.save(img_path)
            Image.fromarray(binary * 255).save(mask_path)
            manifest[split_tag].append({"image": str(img_path), "mask": str(mask_path)})

            print(f"  [{split_tag} {idx+1}/{cap}] {name}", end="\r", flush=True)

            del img, binary
            gc.collect()

        print()  # newline after the \r progress line

    if USE_SAM:
        del sam_gen
        gc.collect()

    manifest_path = DATASET_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    counts = {k: len(v) for k, v in manifest.items()}
    print(f"\nDone. Manifest: {manifest_path}")
    print(f"  train: {counts['train']}  val: {counts['val']}  test: {counts['test']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def prepare_dataset() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    stream_and_save()


if __name__ == "__main__":
    prepare_dataset()
