from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "hybridnets_256x384.onnx"
ANCHOR_PATH = ROOT / "anchors_512x640.npy"


def test_model_file_exists() -> None:
    assert MODEL_PATH.is_file()
    assert MODEL_PATH.stat().st_size > 1_000_000


def test_anchor_file_is_readable() -> None:
    anchors = np.load(ANCHOR_PATH)
    assert anchors.dtype == np.float32
    assert anchors.ndim == 3
    assert anchors.shape[0] == 1
    assert anchors.shape[2] == 4
    assert np.isfinite(anchors).all()


def test_onnxruntime_can_load_model_metadata() -> None:
    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("onnxruntime is not installed")

    import onnxruntime as ort

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    assert session.get_inputs()
    assert session.get_outputs()
