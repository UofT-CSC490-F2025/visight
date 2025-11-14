from __future__ import annotations

import os
import json
import shutil
import math
import cProfile
import pstats
from io import StringIO
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PosixPath
from typing import Optional, Tuple, Union
import yaml
import modal
from ultralytics import YOLO
# Support both local dev (module path: configs.config) and Modal image (file at /config.py)
try:
    from configs.config import Config  # running from repo (e.g., cwd=modal/)
except Exception:
    from config import Config  # running inside Modal image where config.py is at /

# ----------------------------
# Modal app & image
# ----------------------------
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])  # OpenCV deps
    # CUDA-enabled PyTorch for A10G/A100 (CUDA 12.1 wheelss)
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install([
        "ultralytics>=8.3.0",
        "opencv-python~=4.10.0",
        "numpy>=1.24,<2.0",
        "PyYAML",
        "onnx>=1.14.0",
        "tensorboard",
        "future",
    ])
    .env({"YOLO_CONFIG_DIR": "/tmp/Ultralytics"})
    .add_local_file("configs/config.py", remote_path="/root/config.py")
    .add_local_file("configs/model_config.yaml", remote_path="/root/model_config.yaml")
)

app = modal.App("visight-yolo-finetune", image=image)

# Secrets and mounts
s3_secret = modal.Secret.from_name(
    "s3-bucket-secret",
    required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
)

# wandb_secret = modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])
vol = modal.Volume.from_name("visight-yolo-runs", create_if_missing=True)

# ----------------------------
# CONFIG
# ----------------------------
CONFIG = Config()

BUCKET_NAME = CONFIG.bucket_name
ULTRALYTICS_VERSION = CONFIG.ultralytics_version
OPTIONAL_TRAIN_SPEC_FIELDS = ["warmup_epochs", "dropout", "freeze"]
# S3 bucket is mounted inside the container at /bucket (CloudBucketMount)
MOUNT_PATH = Path("/bucket") #PosixPath("/bucket")

VOLUME_PATH = Path("/root/data")
DATA_WORKDIR = VOLUME_PATH / "work"            # where we stage datasets locally
RUNS_DIR = VOLUME_PATH / "runs"                # where Ultralytics writes runs

# ----------------------------
# Data model
# ----------------------------
@dataclass(frozen=True)
class TrainSpec:
    dataset_version: str                 # "raw", "v1", or any s3 prefix 
    model_size: str = "yolov8s.pt"       # base checkpoint
    epochs: int = 20
    img_size: int = 1280
    batch: Union[int, float, str] = 24
    workers: int = 8
    seed: int = 117
    use_wandb: bool = False
    notes: str = ""
    freeze: int = 1
    warmup_epochs: Optional[int] = None
    dropout: Optional[float] = None
    

    def s3_prefix(self) -> str:
        # Allow friendly shorthands
        if self.dataset_version == "raw":
            return "raw/roboflow/v8"
        if self.dataset_version == "v1":
            return "processed/roboflow/v1"
        if self.dataset_version == "augmented":
            return "processed/roboflow/augmented"
        # Or accept an explicit prefix
        return self.dataset_version


# ----------------------------
# Helpers
# ----------------------------
def _now_utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _safe_copy_file(src: Path, dst: Path) -> None:
    """Copies a file; on some mounts utime/permissions are restricted, so fall back to copyfile."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
    except PermissionError:
        shutil.copyfile(src, dst)


def _copy_dir_tree(src_dir: Path, dst_dir: Path) -> None:
    """Robust directory copy (like cp -r) that works across filesystems/mounts."""
    for root, _, files in os.walk(src_dir):
        rel = Path(root).relative_to(src_dir)
        out_root = dst_dir / rel
        out_root.mkdir(parents=True, exist_ok=True)
        for f in files:
            _safe_copy_file(Path(root) / f, out_root / f)


def _resolve_batch_size(batch_value: Union[int, float, str], num_gpus: int) -> int:
    """Ensure batch size is positive and divisible by number of GPUs."""
    default = max(1, num_gpus) * 8
    value = batch_value
    if isinstance(value, str):
        try:
            value = int(value)
        except Exception:
            return default
    if isinstance(value, float):
        return default
    if isinstance(value, int):
        if value <= 0:
            return default
        if value % num_gpus != 0:
            return math.ceil(value / num_gpus) * num_gpus
        return value
    return default


def stage_dataset_from_s3(prefix: str, force_refresh: bool = False) -> Path:
    """
    Stage s3://BUCKET_NAME/{prefix} into a local working directory so that
    Ultralytics can create *.cache files without hitting 'Function not implemented'
    on CloudBucketMount renames.
    
    PROFILED FUNCTION
    """
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        src_root = MOUNT_PATH / prefix
        if not src_root.exists():
            raise FileNotFoundError(f"S3 prefix not found: s3://{BUCKET_NAME}/{prefix}")

        local_root = DATA_WORKDIR / Path(prefix.replace("/", "_"))
        if local_root.exists():
            if force_refresh:
                shutil.rmtree(local_root)
            else:
                print(f"[data] Reusing staged dataset at {local_root}")
                return local_root
        _copy_dir_tree(src_root, local_root)

        data_yaml = local_root / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"Missing data.yaml at {data_yaml}")

        # YOLO expects data.yaml to contain relative paths to train/val/test.
        # If the Roboflow export uses ../train/images, that's fine since we preserved layout.
        return local_root
    finally:
        profiler.disable()
        
        # Save profiling stats
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Top 30 functions
        
        profile_output_dir = RUNS_DIR / "profiling"
        profile_output_dir.mkdir(parents=True, exist_ok=True)
        profile_file = profile_output_dir / f"stage_dataset_from_s3_{_now_utc_stamp()}.txt"
        profile_file.write_text(s.getvalue())
        print(f"\n[PROFILING] stage_dataset_from_s3 profile saved to: {profile_file}")


def _verify_dataset_dirs(dataset_root: Path) -> None:
    """Fail fast if expected train/val(test)/images folders are missing or empty."""
    candidates = [
        dataset_root / "train" / "images",
        dataset_root / "valid" / "images",
        dataset_root / "val" / "images",
        dataset_root / "test" / "images",
    ]
    missing = [p for p in candidates[:3] if not p.exists()]
    if len(missing) >= 2:  # both val and valid missing, or train missing
        raise FileNotFoundError(
            f"Dataset structure not found under {dataset_root}. Expected 'train/valid' (or 'train/val')."
        )
    # Optional: warn if empty
    for p in candidates:
        if p.exists():
            try:
                any_file = next(p.rglob("*.jpg"), None) or next(p.rglob("*.png"), None)
            except StopIteration:
                any_file = None
            if any_file is None:
                print(f"[warn] No images found under: {p}")



def build_explicit_data_config(dataset_root: Path, orig_yaml: Path) -> dict:
    """Return a dict with absolute train/val/test paths for Ultralytics."""
    with open(orig_yaml, "r", encoding="utf-8") as f:
        orig = yaml.safe_load(f) or {}

    train_dir = dataset_root / "train" / "images"
    val_dir = dataset_root / "valid" / "images"
    if not val_dir.exists():
        alt = dataset_root / "val" / "images"
        if alt.exists():
            val_dir = alt
    test_dir = dataset_root / "test" / "images"

    data = {
        "train": str(train_dir.resolve()),
        "val": str(val_dir.resolve()),
        "nc": orig.get("nc"),
        "names": orig.get("names"),
    }
    if test_dir.exists():
        data["test"] = str(test_dir.resolve())
    return data


def export_onnx(best_weights: Path, run_dir: Path, img_size: int) -> Optional[Path]:
    """Exports best.pt to ONNX and returns the ONNX path if produced."""
    try:
        model_best = YOLO(str(best_weights))
        # This writes under run_dir by default
        model_best.export(format="onnx", imgsz=img_size, opset=12, dynamic=True)
        candidate = run_dir / "weights" / "best.onnx"
        if candidate.exists():
            return candidate
        onnx_files = list(run_dir.rglob("*.onnx"))
        return onnx_files[0] if onnx_files else None
    except Exception as e:
        print(f"[warn] ONNX export failed: {e}")
        return None


def write_model_card(
    dst_dir: Path,
    model_id: str,
    spec: TrainSpec,
    artifacts: dict,
    data_yaml_local: Path,
) -> None:
    card = {
        "model_id": model_id,
        "dataset_version": spec.dataset_version,
        "data_yaml": str(data_yaml_local),
        "model_size": spec.model_size,
        "epochs": spec.epochs,
        "img_size": spec.img_size,
        "batch": spec.batch,
        "seed": spec.seed,
        "notes": spec.notes,
        "artifacts": artifacts,
    }
    for c in OPTIONAL_TRAIN_SPEC_FIELDS:
        if getattr(spec, c) is not None: card[c] = getattr(spec, c)
    
    out = dst_dir / "model_card.json"
    out.write_text(json.dumps(card, indent=2), encoding="utf-8")


def copy_training_artifacts_to_s3(
    run_dir: Path,
    model_id: str,
    results_csv_rel: str = "results.csv",
    save_results_csv: bool = True,
    save_plots: bool = False,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """
    Copies artifacts from run_dir to the S3-mounted models/ and stats/ prefixes.
    Returns tuple(best_pt_s3, onnx_s3, results_csv_s3).
    
    PROFILED FUNCTION #5
    """
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        s3_models_root = MOUNT_PATH / "models" / model_id
        s3_stats_root = MOUNT_PATH / "stats" / "training" / model_id
        s3_models_root.mkdir(parents=True, exist_ok=True)
        s3_stats_root.mkdir(parents=True, exist_ok=True)

        best_pt = run_dir / "weights" / "best.pt"
        if not best_pt.exists():
            # fallback to common layout
            best_pt = run_dir.parent / run_dir.name / "weights" / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError("best.pt not found after training.")

        # Copy best.pt
        best_pt_s3 = s3_models_root / "best.pt"
        _safe_copy_file(best_pt, best_pt_s3)

        # ONNX (if present)
        onnx_s3: Optional[Path] = None
        best_onnx = run_dir / "weights" / "best.onnx"
        if not best_onnx.exists():
            # Fallback to scan if layout differs
            onnx_files = list(run_dir.rglob("*.onnx"))
            best_onnx = onnx_files[0] if onnx_files else None
        if best_onnx and best_onnx.exists():
            onnx_s3 = s3_models_root / "best.onnx"
            _safe_copy_file(best_onnx, onnx_s3)

        # results.csv to stats/
        results_csv_s3: Optional[Path] = None
        if save_results_csv:
            results_csv = run_dir / results_csv_rel
            if results_csv.exists():
                results_csv_s3 = s3_stats_root / "results.csv"
                _safe_copy_file(results_csv, results_csv_s3)

        if save_plots:
            for plot in ["labels.jpg", "confusion_matrix.png", "results.png", "P_curve.png", "R_curve.png"]:
                src = run_dir / plot
                if src.exists():
                    _safe_copy_file(src, s3_stats_root / plot)

        return best_pt_s3, onnx_s3, results_csv_s3
    finally:
        profiler.disable()
        
        # Save profiling stats
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Top 30 functions
        
        profile_output_dir = RUNS_DIR / "profiling"
        profile_output_dir.mkdir(parents=True, exist_ok=True)
        profile_file = profile_output_dir / f"copy_training_artifacts_to_s3_{_now_utc_stamp()}.txt"
        profile_file.write_text(s.getvalue())
        print(f"\n[PROFILING] copy_training_artifacts_to_s3 profile saved to: {profile_file}")


# ----------------------------
# Training function
# ----------------------------
@app.function(
    secrets=[s3_secret],# wandb_secret],
    volumes={MOUNT_PATH: modal.CloudBucketMount(BUCKET_NAME, secret=s3_secret), VOLUME_PATH: vol},
    timeout=60 * 60 * 8,   # up to 8h
    cpu=4,
    gpu="A100:3",
)
def train_yolo(
    dataset_version: str = "raw",        # "raw", "v1", or explicit s3 prefix
    model_size: str = "yolov8s.pt",
    epochs: int = 20,
    img_size: int = 640,
    batch: Union[int, float] = 24,
    workers: int = 8,
    seed: int = 117,
    use_wandb: bool = False,
    notes: str = "",
    export_to_onnx: bool = True,
    n_layers_freeze: float = 1,
    warmup_epochs: Optional[int] = None,
    dropout: Optional[float] = None,
    plots: bool = True,
    fast_mode: bool = False,              # skip staging copy; read from S3 mount directly
    fraction: Optional[float] = None,     # optional dataset fraction for quick runs
    refresh_data: bool = False,           # force re-stage dataset from S3
):
    """
    Fine-tune YOLO on a dataset stored in S3 (mounted), staging the data locally to avoid
    cache/rename issues. Saves best.pt (+ optional best.onnx) to s3://{bucket}/models/{model_id}/
    and results.csv to s3://{bucket}/stats/training/{model_id}/.
    
    PROFILED FUNCTION
    """
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        # Set up spec and environment
        spec = TrainSpec(
            dataset_version=dataset_version,
            model_size=model_size,
            epochs=epochs,
            img_size=img_size,
            batch=batch,
            workers=workers,
            seed=seed,
            use_wandb=use_wandb,
            notes=notes,
            freeze=n_layers_freeze,
            warmup_epochs=warmup_epochs,
            dropout=dropout,
        )

        if spec.use_wandb:
            os.environ.setdefault("WANDB_START_METHOD", "thread")
            # import wandb
            # wandb.init(project="visight")
        else:
            os.environ["WANDB_MODE"] = "disabled"

        # Decide data source
        prefix = spec.s3_prefix()
        if fast_mode:
            # Use S3 mount paths directly; disable caching to avoid rename issues
            dataset_root = MOUNT_PATH / prefix
        else:
            # Stage dataset locally (avoid CloudBucketMount rename limitations)
            dataset_root = stage_dataset_from_s3(prefix, force_refresh=refresh_data)
        orig_yaml = dataset_root / "data.yaml"
        if not orig_yaml.exists():
            raise FileNotFoundError(f"Missing data.yaml at {orig_yaml}")

        # Unique run descriptors
        run_id = _now_utc_stamp()
        base_name = Path(spec.model_size).stem
        model_id = f"{spec.dataset_version}-{base_name}-{run_id}"
        run_dir = RUNS_DIR / model_id
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"[train] Using dataset_root: {dataset_root}")
        _verify_dataset_dirs(dataset_root)
        data_config = build_explicit_data_config(dataset_root, orig_yaml)
        # Persist YAML for bookkeeping under the run directory
        data_yaml_record = run_dir / "data.yaml"
        data_yaml_record.write_text(yaml.safe_dump(data_config, sort_keys=False), encoding="utf-8")
        # Write a copy under /tmp for DDP workers (local FS, no volume propagation delays)
        tmp_yaml = Path("/tmp") / f"{model_id}_data.yaml"
        tmp_yaml.write_text(data_yaml_record.read_text(encoding="utf-8"), encoding="utf-8")
        data_yaml_path = tmp_yaml
        print(f"[train] Using data yaml: {data_yaml_path}")

        # Batch handling: ensure divisibility across GPUs
        num_gpus = 3
        batch_size = _resolve_batch_size(spec.batch, num_gpus)
        if batch_size != spec.batch:
            print(f"[train] Adjusted batch size from {spec.batch} to {batch_size} for {num_gpus} GPUs")

        # Train
        model = YOLO(spec.model_size)
        train_kwargs = dict(
            data=str(data_yaml_path),
            imgsz=spec.img_size,
            epochs=spec.epochs,
            device="0,1,2",
            batch=batch_size,
            workers=spec.workers,
            cache=(False if fast_mode else True),
            project=str(RUNS_DIR),
            name=model_id,
            exist_ok=True,
            seed=spec.seed,
            verbose=True,
            plots=plots,
        )
        if fraction is not None:
            train_kwargs["fraction"] = fraction
        train_kwargs.update({k: getattr(spec, k) for k in OPTIONAL_TRAIN_SPEC_FIELDS if getattr(spec, k) is not None})
        # Pass through kwargs directly; Ultralytics train() accepts **kwargs and will ignore unknowns.
        # Previous filtering caused arguments to be dropped on versions exposing only **kwargs in signature.
        model.train(**train_kwargs)

        # Optional: export ONNX
        onnx_path: Optional[Path] = None
        if export_to_onnx:
            onnx_path = export_onnx(run_dir / "weights" / "best.pt", run_dir, spec.img_size)

        # Persist artifacts to S3
        best_pt_s3, onnx_s3, results_csv_s3 = copy_training_artifacts_to_s3(
            run_dir=run_dir,
            model_id=model_id,
            save_results_csv=True,
            save_plots=plots,
        )

        # Model card
        artifacts = {
            "best_pt": f"s3://{BUCKET_NAME}/models/{model_id}/best.pt",
            "best_onnx": f"s3://{BUCKET_NAME}/models/{model_id}/best.onnx" if onnx_s3 else None,
            "results_csv": f"s3://{BUCKET_NAME}/stats/training/{model_id}/results.csv" if results_csv_s3 else None,
        }
        write_model_card(
            dst_dir=run_dir,
            model_id=model_id,
            spec=spec,
            artifacts=artifacts,
            data_yaml_local=data_yaml_path,
        )
        _safe_copy_file(run_dir / "model_card.json", MOUNT_PATH / "models" / model_id / "model_card.json")

        # Persist volume state
        vol.commit()

        print("Training complete.")
        print("Saved artifacts:")
        print("  ", artifacts["best_pt"])
        if artifacts["best_onnx"]:
            print("  ", artifacts["best_onnx"])
        if artifacts["results_csv"]:
            print("  ", artifacts["results_csv"])
        print("Model card:")
        print("  ", f"s3://{BUCKET_NAME}/models/{model_id}/model_card.json")
    finally:
        profiler.disable()
        
        # Save profiling stats
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(50)  # Top 50 functions
        
        profile_output_dir = RUNS_DIR / "profiling"
        profile_output_dir.mkdir(parents=True, exist_ok=True)
        profile_file = profile_output_dir / f"train_yolo_{_now_utc_stamp()}.txt"
        profile_file.write_text(s.getvalue())
        print(f"\n[PROFILING] train_yolo profile saved to: {profile_file}")




# ----------------------------
# Local entrypoint
# ----------------------------
@app.local_entrypoint()
def main(
    params: Optional[str] = "configs/model_config.yaml",
    quick_check: bool = False,
):
    """
    Kick off a training job on Modal using a single source of truth
    for training parameters: the YAML file at `params`.
    """
    # Defaults used only if YAML omits a key
    defaults = {
        "dataset_version": "augmented",
        "model_size": "yolov8s.pt",
        "epochs": 30,
        "img_size": 640,
        "batch": 24,
        "workers": 8,
        "seed": 117,
        "use_wandb": False,
        "notes": "",
        "export_to_onnx": True,
        "warmup_epochs": 3,
        "dropout": 0.0,
        "plots": True,
        "n_layers_freeze": 0,
        "fast_mode": False,
        "fraction": None,
        "refresh_data": False,
    }

    param_dict = dict(defaults)
    # Resolve default params path for local vs remote contexts
    if params and Path(params).exists():
        with open(params, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        for k in param_dict.keys():
            if k in y and y[k] is not None:
                param_dict[k] = y[k]
    else:
        # Try alternative default when running inside container (not typical for local_entrypoint)
        alt_params = "/model_config.yaml"
        if Path(alt_params).exists():
            with open(alt_params, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
            for k in param_dict.keys():
                if k in y and y[k] is not None:
                    param_dict[k] = y[k]
        else:
            print(f"[warn] Params file not found at {params}; using defaults.")

    if quick_check:
        # Make quick runs snappy: 1 epoch, no cache, smaller sample, no plots
        param_dict["epochs"] = 1
        param_dict["fast_mode"] = True
        param_dict["plots"] = False
        param_dict["fraction"] = 0.05 if param_dict.get("fraction") in (None, 0) else param_dict["fraction"]

    train_yolo.remote(**param_dict)
