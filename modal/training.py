from __future__ import annotations

import os
import json
import shutil
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
from configs.config import Config

# ----------------------------
# Modal app & image
# ----------------------------
image = (modal.Image.from_dockerfile("Dockerfile")
    .apt_install(["libgl1", "libglib2.0-0"])
    .add_local_file("./configs/config.py", remote_path="/config.py")
    .add_local_file("./configs/model_config.yaml", remote_path="/model_config.yaml")
)

app = modal.App("visight-yolo-finetune", image=image)

# Secrets and mounts
s3_secret = modal.Secret.from_name(
    "s3-bucket-secret",
    required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
)
s3_secret_backup = modal.Secret.from_name(
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
    batch: float = 0.95 #Union[float, int] = 0.95            # auto-batch target (float) or fixed int
    workers: int = 4
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


def stage_dataset_from_s3(prefix: str) -> Path:
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
        # Always refresh to mirror S3 (small cost, avoids stale state)
        if local_root.exists():
            shutil.rmtree(local_root)
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
    timeout=60 * 60 * 4,   # up to 4h
    cpu=4,
    gpu="A10G:1",
)
def train_yolo(
    dataset_version: str = "raw",        # "raw", "v1", or explicit s3 prefix
    model_size: str = "yolov8s.pt",
    epochs: int = 20,
    img_size: int = 640,
    batch: float = 0.95, #Union[float, int] = 0.95,
    use_wandb: bool = False,
    notes: str = "",
    export_to_onnx: bool = True,
    n_layers_freeze: float = 1,
    warmup_epochs: Optional[int] = None,
    dropout: Optional[float] = None,
    plots: bool = True,
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

        # Stage dataset locally (avoid CloudBucketMount rename limitations)
        prefix = spec.s3_prefix()
        local_data_root = stage_dataset_from_s3(prefix)
        data_yaml_path = local_data_root / "data.yaml"

        # Unique run descriptors
        run_id = _now_utc_stamp()
        base_name = Path(spec.model_size).stem
        model_id = f"{spec.dataset_version}-{base_name}-{run_id}"
        run_dir = RUNS_DIR / model_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Train
        model = YOLO(spec.model_size)
        model.train(
            data=str(data_yaml_path),
            imgsz=spec.img_size,
            epochs=spec.epochs,
            device=0,                # single GPU
            batch=spec.batch,
            workers=spec.workers,
            cache=True,              # enable cache now that data is staged locally
            project=str(RUNS_DIR),
            name=model_id,
            exist_ok=True,
            seed=spec.seed,
            verbose=True,
            plots=plots,
            **{k: getattr(spec, k) for k in OPTIONAL_TRAIN_SPEC_FIELDS if getattr(spec, k) is not None}
        )

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



@app.function(
    secrets=[s3_secret],# wandb_secret],
    volumes={MOUNT_PATH: modal.CloudBucketMount(BUCKET_NAME, secret=s3_secret_backup), VOLUME_PATH: vol},
    timeout=60 * 60 * 4,   # up to 4h
    cpu=4,
    gpu="A10G:1",
)
def train_yolo_backup(
    dataset_version: str = "raw",        # "raw", "v1", or explicit s3 prefix
    model_size: str = "yolov8s.pt",
    epochs: int = 20,
    img_size: int = 640,
    batch: float = 0.95, #Union[float, int] = 0.95,
    use_wandb: bool = False,
    notes: str = "",
    export_to_onnx: bool = True,
    n_layers_freeze: float = 1,
    warmup_epochs: Optional[int] = None,
    dropout: Optional[float] = None,
    plots: bool = True,
):
    """
    Fine-tune YOLO on a dataset stored in S3 (mounted), staging the data locally to avoid
    cache/rename issues. Saves best.pt (+ optional best.onnx) to s3://{bucket}/models/{model_id}/
    and results.csv to s3://{bucket}/stats/training/{model_id}/.
    """
    # Set up spec and environment
    spec = TrainSpec(
        dataset_version=dataset_version,
        model_size=model_size,
        epochs=epochs,
        img_size=img_size,
        batch=batch,
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

    # Stage dataset locally (avoid CloudBucketMount rename limitations)
    prefix = spec.s3_prefix()
    local_data_root = stage_dataset_from_s3(prefix)
    data_yaml_path = local_data_root / "data.yaml"

    # Unique run descriptors
    run_id = _now_utc_stamp()
    base_name = Path(spec.model_size).stem
    model_id = f"{spec.dataset_version}-{base_name}-{run_id}"
    run_dir = RUNS_DIR / model_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Train
    model = YOLO(spec.model_size)
    model.train(
        data=str(data_yaml_path),
        imgsz=spec.img_size,
        epochs=spec.epochs,
        device=0,                # single GPU
        batch=spec.batch,
        workers=spec.workers,
        cache=False,             # keep False; we staged locally anyway
        project=str(RUNS_DIR),
        name=model_id,
        exist_ok=True,
        seed=spec.seed,
        verbose=True,
        plots=plots,
        **{k: getattr(spec, k) for k in OPTIONAL_TRAIN_SPEC_FIELDS if getattr(spec, k) is not None}
    )

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

# ----------------------------
# Local entrypoint
# ----------------------------
@app.local_entrypoint()
def main(
    params: Optional[str] = None,  # Can pass in a config (ex: ./configs/model_config.yaml). This overrides all other args of this function
    dataset_version: str = "raw",   # "raw", "v1", or explicit s3 prefix (e.g., "tmp/smoke_v1")
    model_size: str = "yolov8s.pt",
    epochs: int = 10,
    img_size: int = 640,
    batch: float = 0.95, #Union[float, int] = 0.95,
    use_wandb: bool = False,
    notes: str = "",
    quick_check: bool = False,
    export_to_onnx: bool = True,
    warmup_epochs: int = 0, 
    dropout:float=0.3,
    plots:bool=True,
    freeze:float=1,
):
        
    param_dict = {
        "dataset_version":dataset_version,
        "model_size":model_size,
        "epochs":epochs,
        "img_size":img_size,
        "batch":batch,
        "use_wandb":use_wandb,
        "notes":notes,
        "export_to_onnx":export_to_onnx,
        "warmup_epochs":warmup_epochs, 
        "dropout":dropout,
        "plots":plots, 
        "n_layers_freeze":freeze
    }
    
    if params: 
        with open(params, "r") as param_file: 
            param_override = yaml.safe_load(param_file)
        param_dict = {k:v if param_override.get(k, None) is None else param_override[k] for k,v in param_dict.items()}
    """
    Kick off a training job on Modal.
    Use quick_check=True for a fast dry run (epochs forced to 1).
    """
    if quick_check:
        param_dict["epochs"] = 1
    try: 
        train_yolo.remote(**param_dict)
    except: 
        train_yolo_backup.remote(**param_dict)
