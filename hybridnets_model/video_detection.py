from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "hybridnets_256x384.onnx"
DEFAULT_ANCHORS = ROOT / "anchors_256x384.npy"
DEFAULT_OUTPUT = ROOT / "outputs" / "output_hybridnets.mp4"


def infer_height_width(path: Path) -> tuple[int, int] | None:
    match = re.search(r"(\d+)x(\d+)", path.stem)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def provider_list(ort_module: Any, requested: str) -> list[str]:
    available = ort_module.get_available_providers()
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    if requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider is not available. Available: {available}")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def patch_onnxruntime_session(ort_module: Any, default_providers: list[str]) -> None:
    original_session = ort_module.InferenceSession
    if getattr(original_session, "_hybridnets_cuda_patch", False):
        return

    class PatchedSession(original_session):  # type: ignore[misc, valid-type]
        _hybridnets_cuda_patch = True

        def __init__(
            self,
            model_path: str,
            sess_options: Any = None,
            providers: list[Any] | None = None,
            provider_options: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> None:
            providers = providers or default_providers
            provider_options = provider_options or [{} for _ in providers]

            patched_providers = []
            patched_options = []
            for index, provider in enumerate(providers):
                if isinstance(provider, tuple):
                    name, options = provider
                    options = dict(options)
                else:
                    name = provider
                    options = dict(provider_options[index]) if index < len(provider_options) else {}

                if name == "CUDAExecutionProvider":
                    options.setdefault("cudnn_conv_algo_search", "DEFAULT")

                patched_providers.append(name)
                patched_options.append(options)

            super().__init__(
                model_path,
                sess_options=sess_options,
                providers=patched_providers,
                provider_options=patched_options,
                **kwargs,
            )

    ort_module.InferenceSession = PatchedSession


def import_hybridnets(wrapper_path: Path | None) -> tuple[Any, Any]:
    if wrapper_path is not None:
        sys.path.insert(0, str(wrapper_path.resolve()))

    try:
        from hybridnets import HybridNets, optimized_model
    except ModuleNotFoundError as exc:
        if exc.name != "hybridnets":
            raise
        hint = (
            "Could not import the HybridNets wrapper. Install the project that provides "
            "`hybridnets.py`, or pass its directory with --wrapper-path."
        )
        raise RuntimeError(hint) from exc

    return HybridNets, optimized_model


def valid_file(path: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {resolved}")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HybridNets ONNX inference on a video.")
    parser.add_argument("--video", required=True, type=valid_file, help="Input video path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, type=valid_file, help="ONNX model path.")
    parser.add_argument("--anchors", default=DEFAULT_ANCHORS, type=valid_file, help="Anchor .npy path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path, help="Output video path.")
    parser.add_argument(
        "--wrapper-path",
        default=os.environ.get("HYBRIDNETS_WRAPPER_PATH"),
        type=Path,
        help="Directory containing the external hybridnets.py wrapper.",
    )
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        help="Frame resize size passed to the model. Defaults to dimensions inferred from model name.",
    )
    parser.add_argument("--conf-thres", default=0.5, type=float, help="Detection confidence threshold.")
    parser.add_argument("--iou-thres", default=0.5, type=float, help="Detection IOU threshold.")
    parser.add_argument(
        "--provider",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="ONNX Runtime execution provider.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run the wrapper's optimized_model(model_path) before inference.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    import cv2
    import onnxruntime as ort

    selected_providers = provider_list(ort, args.provider)
    patch_onnxruntime_session(ort, selected_providers)

    model_hw = infer_height_width(args.model)
    anchor_hw = infer_height_width(args.anchors)
    if model_hw and anchor_hw and model_hw != anchor_hw:
        print(
            "Warning: model filename implies "
            f"{model_hw[0]}x{model_hw[1]}, but anchor filename implies "
            f"{anchor_hw[0]}x{anchor_hw[1]}."
        )

    if args.input_size:
        input_size = tuple(args.input_size)
    elif model_hw:
        input_size = (model_hw[1], model_hw[0])
    else:
        input_size = (640, 512)

    HybridNets, optimized_model = import_hybridnets(args.wrapper_path)

    print("Available providers:", ort.get_available_providers())
    print("Selected providers:", selected_providers)
    print("Input size:", input_size)

    if args.optimize:
        optimized_model(str(args.model))

    road_estimator = HybridNets(
        str(args.model),
        str(args.anchors),
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
    )

    session = getattr(road_estimator, "session", None)
    if session is not None:
        print("Active providers:", session.get_providers())

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if original_width <= 0 or original_height <= 0:
        raise RuntimeError("Input video reported invalid dimensions.")
    if fps <= 0:
        fps = 30.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(args.output), fourcc, fps, (original_width, original_height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {args.output}")

    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            input_frame = cv2.resize(frame, input_size)
            road_estimator(input_frame)
            result = road_estimator.draw_2D(input_frame)
            result = cv2.resize(result, (original_width, original_height))
            out.write(result)
    finally:
        cap.release()
        out.release()

    print(f"Done. Processed {frame_count} frames -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
