import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_data_yaml(data_yaml_path: Path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    root = data_yaml_path.parent
    paths = {}
    for key in ["train", "val", "test"]:
        if key in y:
            # Roboflow often uses "../train/images" etc.
            rel = Path(y[key])
            rel_str = str(rel)
            if rel_str.startswith("../"):
                rel = Path(rel_str.replace("../", "", 1))
            paths[key] = (root / rel).resolve()

    names = y["names"]
    nc = y["nc"]
    assert len(names) == nc, "names length must equal nc"
    return paths, names

def read_yolo_label_classes(label_path: Path):
    s = set()
    if not label_path.exists():
        return s
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cid = int(parts[0])
                s.add(cid)
            except (ValueError, IndexError):
                pass
    return s

def count_split(images_dir: Path, labels_dir: Path):
    per_class = Counter()
    img_count = 0
    for p in images_dir.rglob("*"):
        if p.is_dir() or p.suffix.lower() not in IMG_EXTS:
            continue
        img_count += 1
        lbl = labels_dir / f"{p.stem}.txt"
        classes = read_yolo_label_classes(lbl)
        for c in classes:
            per_class[c] += 1
    return img_count, per_class

def main():
    ap = argparse.ArgumentParser("Count per-class coverage in Roboflow splits")
    ap.add_argument("--root", required=True, help="Path to 'F1 Logos' folder containing data.yaml, train/, valid/, test/")
    ap.add_argument("--out", default="./roboflow_split_counts", help="Output directory for JSON/CSV")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = Path(args.out).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    data_yaml = root / "data.yaml"
    paths, names = load_data_yaml(data_yaml)
    nc = len(names)

    # Map 'val' -> 'valid' on disk if needed
    def images_labels_from(images_path: Path):
        # images_path like ".../train/images"
        base = images_path.parent  # ".../train"
        labels = base / "labels"
        return images_path, labels

    results = {}
    totals = {}
    for key in ["train", "val", "test"]:
        images_path = paths.get(key)
        if images_path is None:
            continue
        imgs_dir, lbls_dir = images_labels_from(images_path)
        if not imgs_dir.exists():
            # Try alternate Roboflow naming (e.g., "valid" instead of "val")
            # If data.yaml pointed to ".../val/images" but on disk it is ".../valid/images"
            alt = images_path.parent.parent / ("valid" if key == "val" else key) / "images"
            if alt.exists():
                imgs_dir = alt
                lbls_dir = alt.parent / "labels"
        img_count, per_class = count_split(imgs_dir, lbls_dir)
        totals[key] = img_count
        results[key] = {i: per_class.get(i, 0) for i in range(nc)}

    # Write JSON
    with open(outdir / "class_counts_by_split.json", "w", encoding="utf-8") as f:
        json.dump({
            "names": names,
            "totals": totals,
            "counts": results
        }, f, indent=2)

    # Write CSV (wide format: one row per class)
    with open(outdir / "class_counts_by_split.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "train", "val", "test"])
        for i, cname in enumerate(names):
            train_c = results.get("train", {}).get(i, 0)
            val_c   = results.get("val", {}).get(i, 0)
            test_c  = results.get("test", {}).get(i, 0)
            w.writerow([i, cname, train_c, val_c, test_c])

    # Pretty print a small summary
    print("\nSplit totals (images with any labels):")
    for k, v in totals.items():
        print(f"  {k}: {v}")

    rare_in_val = [i for i in range(nc) if results.get("val", {}).get(i, 0) == 0]
    rare_in_test = [i for i in range(nc) if results.get("test", {}).get(i, 0) == 0]
    if rare_in_val or rare_in_test:
        print("\nClasses missing from validation or test (by id:name):")
        if rare_in_val:
            print("  Missing in val:", [(i, names[i]) for i in rare_in_val])
        if rare_in_test:
            print("  Missing in test:", [(i, names[i]) for i in rare_in_test])

    print(f"\nWrote:\n- {outdir/'class_counts_by_split.json'}\n- {outdir/'class_counts_by_split.csv'}")

if __name__ == "__main__":
    main()