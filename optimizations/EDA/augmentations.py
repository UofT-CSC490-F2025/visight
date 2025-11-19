import argparse
import os
import random
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
import yaml

def build_augment_pipeline():
    return A.Compose([
        # ---- photometric ----
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=20, p=0.5),

        # ---- blur family  ----
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.35),

        # ---- compression / low-res ----
        A.OneOf([
            A.ImageCompression(quality_range=(50, 90), p=1.0),
            A.Downscale(scale=0.5, p=1.0),  # simulate broadcast/downsampled frames
        ], p=0.5),

        # ---- noise family  ----
        A.OneOf([
            A.GaussNoise(std_range=(0.1, 0.2), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.4), p=1.0),
        ], p=0.3),

        # ---- mild geometry ----
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.08),
                 rotate=(-5, 5), shear=(-3, 3), p=0.7),
        A.Perspective(scale=(0.005, 0.02), p=0.15),

        # ---- occasional bbox-safe crop to upsample small logos ----
        A.RandomScale(scale_limit=0.15, p=0.25),  

        # ---- light occlusion (rare) ----
        A.CoarseDropout(num_holes_range=(1, 1),
                       hole_height_range=(0.06, 0.12),
                       hole_width_range=(0.06, 0.12),
                       fill=0,
                       p=0.15),
    ],
    bbox_params=A.BboxParams(format='yolo',
                             label_fields=['class_labels'],
                             min_visibility=0.2))


def read_yolo_labels(label_path: Path):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            boxes.append((cls, x, y, w, h))
    return boxes

def write_yolo_labels(label_path: Path, boxes):
    with open(label_path, "w") as f:
        for cls, x, y, w, h in boxes:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def img_extensions():
    return {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def compute_class_counts(label_dir: Path):
    counts = Counter()
    for lp in sorted(label_dir.glob("*.txt")):
        for (cls, *_rest) in read_yolo_labels(lp):
            counts[cls] += 1
    return counts

def image_label_pairs(images_dir: Path, labels_dir: Path):
    pairs = []
    for img_path in images_dir.glob("*"):
        if img_path.suffix.lower() not in img_extensions():
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        pairs.append((img_path, label_path))
    return pairs

def resolve_path(base_dir: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (base_dir / q)

def main():
    parser = argparse.ArgumentParser(description="Offline long-tail augmentation for YOLO datasets (Ultralytics layout)")
    parser.add_argument("--data_yaml", type=str, required=True, help="Path to data.yaml (Ultralytics style)")
    parser.add_argument("--floor", type=int, default=150, help="Minimum train instances per class after augmentation")
    parser.add_argument("--max_per_image", type=int, default=3, help="Max augmented copies per source image per pass")
    parser.add_argument("--suffix", type=str, default="_aug", help="Suffix for augmented files")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on total augmented samples to generate (0=unlimited)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_yaml_path = Path(args.data_yaml).resolve()
    data = load_yaml(data_yaml_path)
    base_dir = data_yaml_path.parent

    train_spec = str(data["train"])
    train_path = resolve_path(base_dir, train_spec)

    if train_path.name.lower() == "images":
        images_dir = train_path
        labels_dir = train_path.parent / "labels"
    else:
        images_dir = train_path / "images"
        labels_dir = train_path / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Could not find expected train/images and train/labels under {train_path} "
                                f"(resolved from '{train_spec}' relative to {base_dir})")

    print(f"[INFO] Using images: {images_dir}")
    print(f"[INFO] Using labels:  {labels_dir}")

    counts = compute_class_counts(labels_dir)
    if not counts:
        print("[WARN] No labels found in train/labels. Nothing to augment.")
        return

    all_classes = sorted(counts.keys())
    print("\n[BEFORE] Train instances per class:")
    for c in all_classes:
        print(f"  class {c}: {counts[c]}")

    pairs = image_label_pairs(images_dir, labels_dir)
    class_to_images = defaultdict(list)
    for idx, (img_path, lbl_path) in enumerate(pairs):
        boxes = read_yolo_labels(lbl_path)
        if not boxes:
            continue
        for c in set(b[0] for b in boxes):
            class_to_images[c].append(idx)

    augment = build_augment_pipeline()

    total_created = 0
    progress = True
    pbar = tqdm(total=1, desc="Augmenting", dynamic_ncols=True)
    while progress:
        progress = False
        for c in all_classes:
            current = counts[c]
            if current >= args.floor:
                continue
            need = args.floor - current
            candidates = class_to_images.get(c, [])
            if not candidates:
                continue

            created_for_c = 0
            random.shuffle(candidates)

            for idx in candidates:
                if created_for_c >= need:
                    break
                img_path, lbl_path = pairs[idx]
                boxes = read_yolo_labels(lbl_path)
                if not boxes:
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                bboxes = []
                labels = []
                for (cls_id, x, y, bw, bh) in boxes:
                    bboxes.append([x, y, bw, bh])
                    labels.append(cls_id)

                try:
                    transformed = augment(image=img, bboxes=bboxes, class_labels=labels)
                except Exception:
                    continue

                out_img = transformed["image"]
                out_bboxes = transformed["bboxes"]
                out_labels = transformed["class_labels"]

                if len(out_bboxes) == 0:
                    continue

                base = img_path.stem
                out_name = f"{base}{args.suffix}_{c}_{created_for_c}"
                out_img_path = img_path.parent / f"{out_name}{img_path.suffix}"
                out_lbl_path = lbl_path.parent / f"{out_name}.txt"

                if out_img_path.exists() or out_lbl_path.exists():
                    continue

                cv2.imwrite(str(out_img_path), out_img)

                new_boxes = []
                for lab, (x, y, bw, bh) in zip(out_labels, out_bboxes):
                    x = float(np.clip(x, 0.0, 1.0))
                    y = float(np.clip(y, 0.0, 1.0))
                    bw = float(np.clip(bw, 0.0, 1.0))
                    bh = float(np.clip(bh, 0.0, 1.0))
                    if bw <= 0 or bh <= 0:
                        continue
                    new_boxes.append((int(lab), x, y, bw, bh))

                if not new_boxes:
                    try:
                        out_img_path.unlink(missing_ok=True)
                    except TypeError:
                        if out_img_path.exists():
                            os.remove(out_img_path)
                    continue

                write_yolo_labels(out_lbl_path, new_boxes)

                # Update counts for classes present in this augmentation
                per_class_inc = Counter([b[0] for b in new_boxes])
                for pc, inc in per_class_inc.items():
                    counts[pc] += inc

                created_for_c += 1
                total_created += 1
                progress = True
                pbar.update(0)

                if args.limit > 0 and total_created >= args.limit:
                    progress = False
                    break

                if created_for_c >= args.max_per_image:
                    break

            if args.limit > 0 and total_created >= args.limit:
                break

        if args.limit > 0 and total_created >= args.limit:
            break

        if not progress:
            break

    pbar.close()

    print("\n[AFTER] Train instances per class:")
    for c in all_classes:
        print(f"  class {c}: {counts[c]}")

    print(f"\n[SUMMARY] Created {total_created} augmented images.")
    print("Augmentation complete.")

if __name__ == "__main__":
    main()