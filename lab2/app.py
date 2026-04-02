"""
Flask + Waitress inference server for the aerial building segmentation model.

Endpoints:
    GET  /health          → {"status": "ok", "model_loaded": bool}
    POST /predict         → multipart/form-data with field "image" (PNG/JPEG)
                            optional field "mask" (ground-truth PNG, for IoU/Dice)
                         ← {"mask_base64": str, "building_coverage": float,
                             "iou": float|null, "dice": float|null}

Secrets loaded from .env via python-dotenv (never hard-coded).
"""

import os

# Prevent PyTorch/MKL deadlocks before any heavy imports
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import io
import base64
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from flask import Flask, request, jsonify
from waitress import serve
from dotenv import load_dotenv

load_dotenv()

# ── Secrets / config ─────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "weights/segmentation_model.pth")
PORT       = int(os.getenv("PORT", 5000))
IMG_SIZE   = 256

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model helpers ─────────────────────────────────────────────────────────────

def _build_model_arch() -> nn.Module:
    model = deeplabv3_resnet50(weights=None)
    model.classifier[4]     = nn.Conv2d(256, 2, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    return model


def load_model(model_path: str) -> nn.Module | None:
    path = Path(model_path)
    if not path.exists():
        print(f"[WARNING] Model weights not found at '{model_path}'. "
              "Run train.py first. /predict will return 503 until weights are present.")
        return None
    model = _build_model_arch()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    print(f"[INFO] Model loaded from {model_path}")
    return model


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    p, t         = pred.astype(bool), target.astype(bool)
    intersection = float((p & t).sum())
    union        = float((p | t).sum())
    return intersection / (union + 1e-8)


def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    p, t         = pred.astype(bool), target.astype(bool)
    intersection = float((p & t).sum())
    return 2.0 * intersection / (float(p.sum()) + float(t.sum()) + 1e-8)


# ── App factory (enables easy testing without loading real weights) ────────────

def create_app(model_override=None) -> Flask:
    app   = Flask(__name__)
    _model = model_override if model_override is not None else load_model(MODEL_PATH)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model_loaded": _model is not None})

    @app.route("/predict", methods=["POST"])
    def predict():
        if _model is None:
            return jsonify({"error": "Model not loaded. Run train.py first."}), 503

        if "image" not in request.files:
            return jsonify({"error": "No 'image' field in request."}), 400

        try:
            # ── Read input image ─────────────────────────────────────────────
            img_file = request.files["image"]
            img      = Image.open(img_file.stream).convert("RGB")
            orig_w, orig_h = img.size

            # ── Inference ────────────────────────────────────────────────────
            tensor = IMG_TRANSFORM(img).unsqueeze(0)   # [1, 3, H, W]
            with torch.no_grad():
                output = _model(tensor)["out"]          # [1, 2, H, W]

            pred_mask = output.argmax(dim=1).squeeze(0).numpy().astype(np.uint8)  # [H, W]

            # ── Encode mask as base64 PNG (resized to original resolution) ───
            mask_pil = Image.fromarray(pred_mask * 255).resize(
                (orig_w, orig_h), Image.NEAREST
            )
            buf = io.BytesIO()
            mask_pil.save(buf, format="PNG")
            mask_b64 = base64.b64encode(buf.getvalue()).decode()

            building_coverage = float(pred_mask.mean())

            # ── Optional: IoU / Dice against provided ground-truth mask ──────
            iou_val  = None
            dice_val = None
            if "mask" in request.files:
                gt_file = request.files["mask"]
                gt_mask = Image.open(gt_file.stream).convert("L").resize(
                    (IMG_SIZE, IMG_SIZE), Image.NEAREST
                )
                gt_arr  = (np.array(gt_mask) > 127).astype(np.uint8)
                iou_val  = round(compute_iou(pred_mask,  gt_arr), 4)
                dice_val = round(compute_dice(pred_mask, gt_arr), 4)

            return jsonify({
                "mask_base64":       mask_b64,
                "building_coverage": round(building_coverage, 4),
                "iou":               iou_val,
                "dice":              dice_val,
            })

        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    print(f"Starting server on port {PORT}...", flush=True)
    serve(app, host="0.0.0.0", port=PORT, threads=4)
