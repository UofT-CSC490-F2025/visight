"""
Convert a trained YOLO .pt checkpoint to TensorRT (engine) inside Modal.

Usage (from repo root):
    modal run modal/pt_tensorRT.py --latest-run <run_name> --imgsz 1280 --half true

Expects weights at s3://visight-data-yusufmoola/models/<run_name>/best.pt
and writes best.engine (and best.onnx if requested) back to the same folder.
"""

from __future__ import annotations

import modal
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

try:
    from configs.config import Config  # running from repo (e.g., cwd=modal/)
except Exception:
    from config import Config 


CONFIG = Config()
BUCKET_NAME = CONFIG.bucket_name
MOUNT_PATH = Path("/bucket")

# Image with torch + ultralytics + onnx; TensorRT must be available in the runtime for engine export.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install([
        CONFIG.ultralytics_version,
        "onnx>=1.14.0",
        "numpy>=1.24,<2.0",
    ])
    .add_local_file("configs/config.py", remote_path="/root/config.py")
)

app = modal.App("visight-pt-to-trt", image=image)

s3_secret = modal.Secret.from_name(
    "s3-bucket-secret",
    required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
)


@dataclass
class ConvertSpec:
    latest_run: str
    imgsz: int = 640
    half: bool = True
    dynamic: bool = False
    export_onnx: bool = False


@app.function(
    secrets=[s3_secret],
    volumes={MOUNT_PATH: modal.CloudBucketMount(BUCKET_NAME, secret=s3_secret)},
    gpu="A100:1",
    timeout=60 * 60,
)
def convert_to_trt(spec: ConvertSpec):
    from ultralytics import YOLO

    model_dir = MOUNT_PATH / "models" / spec.latest_run
    pt_path = model_dir / "best.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"best.pt not found at {pt_path}")

    print(f"[convert] Loading {pt_path}")
    model = YOLO(str(pt_path))

    # Optional ONNX export
    if spec.export_onnx:
        try:
            model.export(
                format="onnx",
                imgsz=spec.imgsz,
                dynamic=spec.dynamic,
            )
            print(f"[convert] ONNX export complete (in {model_dir})")
        except Exception as e:
            print(f"[convert][warn] ONNX export failed: {e}")

    # TensorRT export
    engine_path: Optional[Path] = None
    try:
        model.export(
            format="engine",
            imgsz=spec.imgsz,
            dynamic=spec.dynamic,
            half=spec.half,
        )
        # Ultralytics writes to the weights directory under the current run
        candidate = model_dir / "weights" / "best.engine"
        if candidate.exists():
            engine_path = candidate
        else:
            # fallback scan
            for cand in model_dir.rglob("*.engine"):
                engine_path = cand
                break
        if engine_path and engine_path.exists():
            print(f"[convert] TensorRT engine written to {engine_path}")
        else:
            print("[convert][warn] Engine export reported success but no .engine file found")
    except Exception as e:
        print(f"[convert][error] TensorRT export failed: {e}")
        raise

    return {
        "pt": str(pt_path),
        "engine": str(engine_path) if engine_path else None,
        "onnx": str(model_dir / "weights" / "best.onnx"),
    }


@app.local_entrypoint()
def main(
    latest_run: str,
    imgsz: int = 640,
    half: bool = True,
    dynamic: bool = False,
    export_onnx: bool = False,
):
    """
    Launch conversion remotely. Example:
        modal run modal/pt_tensorRT.py --latest-run augmented-yolov8m-... --imgsz 1280 --half true
    """
    spec = ConvertSpec(
        latest_run=latest_run,
        imgsz=imgsz,
        half=half,
        dynamic=dynamic,
        export_onnx=export_onnx,
    )
    convert_to_trt.remote(spec)
