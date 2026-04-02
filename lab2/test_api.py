"""
Unit tests for the segmentation Flask API.

Runs without real model weights: a MagicMock replaces the model so
tests can execute in CI before training has happened.

Run:
    pytest test_api.py -v
"""

import io
import json
import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_image_bytes(size=(64, 64)) -> io.BytesIO:
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf


def _make_mask_bytes(size=(64, 64)) -> io.BytesIO:
    arr = np.zeros(size, dtype=np.uint8)
    arr[10:40, 10:40] = 255   # a building-ish square
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf


def _mock_model():
    """Return a MagicMock that mimics model(tensor) → {'out': Tensor[1,2,256,256]}."""
    model = MagicMock()
    out   = torch.zeros(1, 2, 256, 256)
    out[0, 1, 50:150, 50:150] = 5.0   # high logit → class 1 (building) in centre
    model.return_value = {"out": out}
    return model


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Flask test client backed by a mock model (no real weights needed)."""
    from app import create_app
    flask_app = create_app(model_override=_mock_model())
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_ok(self, client):
        data = resp = client.get("/health").get_json()
        assert data["status"] == "ok"

    def test_model_loaded_true(self, client):
        data = client.get("/health").get_json()
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    def test_no_image_returns_400(self, client):
        resp = client.post("/predict")
        assert resp.status_code == 400

    def test_no_image_error_message(self, client):
        data = client.post("/predict").get_json()
        assert "error" in data

    def test_valid_image_returns_200(self, client):
        resp = client.post(
            "/predict",
            data={"image": (self._img(), "aerial.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200

    def test_response_has_mask_base64(self, client):
        data = self._predict(client)
        assert "mask_base64" in data
        assert isinstance(data["mask_base64"], str)
        assert len(data["mask_base64"]) > 0

    def test_response_has_building_coverage(self, client):
        data = self._predict(client)
        assert "building_coverage" in data
        assert 0.0 <= data["building_coverage"] <= 1.0

    def test_iou_none_without_ground_truth(self, client):
        data = self._predict(client)
        assert data["iou"] is None

    def test_dice_none_without_ground_truth(self, client):
        data = self._predict(client)
        assert data["dice"] is None

    def test_iou_and_dice_with_ground_truth(self, client):
        resp = client.post(
            "/predict",
            data={
                "image": (self._img(), "aerial.png"),
                "mask":  (self._mask(), "gt.png"),
            },
            content_type="multipart/form-data",
        )
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["iou"]  is not None
        assert data["dice"] is not None
        assert 0.0 <= data["iou"]  <= 1.0
        assert 0.0 <= data["dice"] <= 1.0

    def test_mask_base64_is_valid_png(self, client):
        import base64
        data = self._predict(client)
        raw  = base64.b64decode(data["mask_base64"])
        img  = Image.open(io.BytesIO(raw))
        assert img.mode in ("L", "P", "RGB", "RGBA")

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _img():
        return _make_image_bytes()

    @staticmethod
    def _mask():
        return _make_mask_bytes()

    def _predict(self, client):
        resp = client.post(
            "/predict",
            data={"image": (self._img(), "aerial.png")},
            content_type="multipart/form-data",
        )
        return resp.get_json()


class TestNoModelLoaded:
    """Verify the 503 path when weights are missing."""

    @pytest.fixture
    def no_model_client(self):
        from app import create_app
        flask_app = create_app(model_override=None)
        flask_app.config["TESTING"] = True
        with flask_app.test_client() as c:
            yield c

    def test_health_model_loaded_false(self, no_model_client):
        data = no_model_client.get("/health").get_json()
        assert data["model_loaded"] is False

    def test_predict_returns_503(self, no_model_client):
        resp = no_model_client.post(
            "/predict",
            data={"image": (_make_image_bytes(), "img.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 503
