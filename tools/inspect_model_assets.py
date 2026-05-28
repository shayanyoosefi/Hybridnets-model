from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "hybridnets_256x384.onnx"
DEFAULT_ANCHORS = ROOT / "anchors_512x640.npy"


def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def inspect_anchors(path: Path) -> None:
    anchors = np.load(path)
    print(f"anchors: {path.name}")
    print(f"  size: {size_mb(path):.2f} MB")
    print(f"  shape: {anchors.shape}")
    print(f"  dtype: {anchors.dtype}")
    print(f"  finite: {bool(np.isfinite(anchors).all())}")
    print(f"  range: {float(anchors.min()):.4f} to {float(anchors.max()):.4f}")


def inspect_model(path: Path) -> None:
    print(f"model: {path.name}")
    print(f"  size: {size_mb(path):.2f} MB")

    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        print("  onnxruntime: not installed; skipping model metadata load")
        return

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print("  providers:", session.get_providers())
    print("  inputs:")
    for value in session.get_inputs():
        print(f"    - {value.name}: shape={value.shape}, type={value.type}")
    print("  outputs:")
    for value in session.get_outputs():
        print(f"    - {value.name}: shape={value.shape}, type={value.type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect local HybridNets model assets.")
    parser.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    parser.add_argument("--anchors", default=DEFAULT_ANCHORS, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not args.anchors.is_file():
        raise FileNotFoundError(args.anchors)

    inspect_model(args.model)
    inspect_anchors(args.anchors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
