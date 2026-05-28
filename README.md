# HybridNets Model Test Repo

This repository contains a local HybridNets ONNX model smoke-test setup.

Current model assets:

- `hybridnets_256x384.onnx`
- `anchors_256x384.npy`

## Setup

Create a virtual environment and install CPU test dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA execution, use `requirements-gpu.txt` instead of `requirements.txt`. Do not install both `onnxruntime` and `onnxruntime-gpu` in the same environment unless you know you need that layout.

## HybridNets Wrapper

Install the external HybridNets wrapper used by the video runner:

```bash
cd /workspace
git clone https://github.com/ibaiGorordo/ONNX-HybridNets-Multitask-Road-Detection.git
python -m pip install -r ONNX-HybridNets-Multitask-Road-Detection/requirements.txt
```

## Smoke Tests

Run the test suite:

```bash
pytest -q
```

The tests validate that the local model and anchor files are present, that the anchors are readable, and, when ONNX Runtime is installed, that the ONNX model can be loaded for metadata inspection.

Inspect the assets directly:

```bash
python tools/inspect_model_assets.py
```

## Video Inference

The video runner expects the external HybridNets Python wrapper that provides:

```python
from hybridnets import HybridNets, optimized_model
```

If that wrapper is not installed as a package, pass its directory with `--wrapper-path`:

```bash
python video-road-detection.py \
  --video path/to/input.mp4 \
  --model hybridnets_256x384.onnx \
  --anchors anchors_256x384.npy \
  --wrapper-path path/to/ONNX-HybridNets-Multitask-Road-Detection \
  --output outputs/output_hybridnets.mp4
```

The script infers resize dimensions from the model filename when it can. For `hybridnets_256x384.onnx`, it resizes frames to width `384` and height `256` unless `--input-size WIDTH HEIGHT` is provided.
