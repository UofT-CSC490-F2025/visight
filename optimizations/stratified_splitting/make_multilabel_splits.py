import argparse
import yaml              
import json
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
import random
import csv


# ----------------------------
## HOW TO RUN THIS SCRIPT
## from the folder that contains "F1 Logos/"
## python3 make_multilabel_splits.py \
 # --root "./F1 Logos" \
 # --out  "./F1 Logos v1" \
 # --ratios "0.75,0.15,0.10" \
 # --seed 42 \
 # --copy

# ----------------------------

# ----------------------------
# Helpers
# ----------------------------

def sha1_stem(path: Path) -> str:
    return hashlib.sha1(path.stem.encode()).hexdigest()

def load_data_yaml(data_yaml_path: Path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    root = data_yaml_path.parent
    paths = {}
    for key in ["train", "val", "test"]:
        if key in y:
            rel = Path(y[key])
            # Handle Roboflow-style "../train/images"
            rel_str = str(rel)
            if rel_str.startswith("../"):
                rel = Path(rel_str.replace("../", "", 1))
            if rel.is_absolute():
                paths[key] = rel
            else:
                paths[key] = (root / rel).resolve()

    names = y["names"]
    nc = y["nc"]
    assert len(names) == nc, "names length must equal nc"
    return paths, names

def read_yolo_labels(label_path: Path):
    """
    Returns set of class_ids present in the image from YOLO txt.
    An empty or missing file returns empty set (treated as background-only).
    """
    classes = set()
    if not label_path.exists():
        return classes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cid = int(parts[0])
                classes.add(cid)
            except (ValueError, IndexError):
                continue
    return classes

def collect_items(split_images_dir: Path, split_labels_dir: Path):
    """
    Returns list of dicts with fields:
      - img_path, lbl_path, stem, id (sha1 of stem), classes (set[int])
    """
    items = []
    for img_path in split_images_dir.rglob("*"):
        if img_path.is_dir():
            continue
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            continue
        stem = img_path.stem
        lbl_path = split_labels_dir / f"{stem}.txt"
        classes = read_yolo_labels(lbl_path)
        items.append({
            "img_path": img_path,
            "lbl_path": lbl_path,
            "stem": stem,
            "id": sha1_stem(img_path),
            "classes": classes
        })
    return items

def pretty_counts_by_split(items_by_split, num_classes):
    out = {}
    for split, items in items_by_split.items():
        c = Counter()
        for it in items:
            for cid in it["classes"]:
                c[cid] += 1
        out[split] = {cid: c.get(cid, 0) for cid in range(num_classes)}
    return out

# ----------------------------
# Greedy iterative multi-label stratification
# ----------------------------

def stratify_greedy(items, num_classes, ratios=(0.75, 0.15, 0.10), seed=42):
    """
    items: list of dicts with field 'classes' (set of class ids)
    Returns: split_assignment dict[id] -> 'train'|'val'|'test'
    """
    random.seed(seed)
    N = len(items)
    splits = ["train", "val", "test"]
    targets = [int(ratios[0]*N), int(ratios[1]*N)]
    targets.append(N - targets[0] - targets[1])  # ensure sum N

    # Desired per-class image counts by split
    class_img_counts = Counter()
    for it in items:
        for c in it["classes"]:
            class_img_counts[c] += 1

    desired = {s: {c: int(class_img_counts[c]*r) for c in range(num_classes)} for s, r in zip(splits, ratios)}

    # Sort classes rarest -> most frequent
    classes_sorted = sorted(range(num_classes), key=lambda c: class_img_counts[c])

    # For assignment bookkeeping
    assign = {}
    remaining = set([it["id"] for it in items])
    by_id = {it["id"]: it for it in items}

    # Current counts
    cur_counts = {s: Counter() for s in splits}
    cur_sizes  = {s: 0 for s in splits}

    # Build inverted index: class -> set(item_ids)
    idx = defaultdict(set)
    for it in items:
        for c in it["classes"]:
            idx[c].add(it["id"])

    # Greedy pass over classes
    for c in classes_sorted:
        c_items = [x for x in idx[c] if x in remaining]
        random.shuffle(c_items)
        for iid in c_items:
            it = by_id[iid]
            # Score each split: deficit for this class + size deficit
            best_split = None
            best_score = None
            for s in splits:
                class_def = desired[s][c] - cur_counts[s][c]
                size_def  = targets[splits.index(s)] - cur_sizes[s]
                # prefer splits that need this class and have capacity
                score = (class_def, size_def)
                if best_score is None or score > best_score:
                    best_score = score
                    best_split = s
            # Assign
            assign[iid] = best_split
            cur_sizes[best_split] += 1
            for cc in it["classes"]:
                cur_counts[best_split][cc] += 1
            remaining.discard(iid)

    # Any leftover items (those with no labels or not touched) -> fill by size
    leftovers = list(remaining)
    random.shuffle(leftovers)
    for iid in leftovers:
        # choose split with most remaining capacity
        best_split = max(splits, key=lambda s: targets[splits.index(s)] - cur_sizes[s])
        assign[iid] = best_split
        cur_sizes[best_split] += 1
        for cc in by_id[iid]["classes"]:
            cur_counts[best_split][cc] += 1

    return assign

# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser("Create multi-label stratified splits (v1) from Roboflow dataset")
    p.add_argument("--root", required=True, help="Path to 'F1 Logos' folder containing data.yaml, train/, valid/, test/")
    p.add_argument("--out",  required=True, help="Output dir for processed split (e.g., ./processed/roboflow/v1)")
    p.add_argument("--ratios", default="0.75,0.15,0.10", help="train,val,test ratios")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true", help="If set, copy files into out/{train,val,test}/{images,labels}")
    args = p.parse_args()

    root = Path(args.root).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    ratios = tuple(float(x) for x in args.ratios.split(","))
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"

    # Load YAML and collect items from ALL available splits
    paths, names = load_data_yaml(root / "data.yaml")
    num_classes = len(names)

    all_items = []
    for split_key in ["train", "val", "test"]:  
        img_dir = paths.get(split_key)
        if img_dir is None:
            continue
        base = img_dir.parent                  
        lbl_dir = base / "labels"               
        if not img_dir.exists():
            continue
        items = collect_items(img_dir, lbl_dir)
        all_items.extend(items)

    print(f"Discovered {len(all_items)} images total across Roboflow splits.")

    # Make stratified assignment
    assignment = stratify_greedy(all_items, num_classes=num_classes, ratios=ratios, seed=args.seed)

    # Produce manifest CSV
    manifest_path = out_root / "split_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","split","img_path","lbl_path","classes"])
        for it in all_items:
            w.writerow([
                it["id"],
                assignment[it["id"]],
                str(it["img_path"]),
                str(it["lbl_path"]),
                json.dumps(sorted(list(it["classes"])))
            ])
    print(f"Wrote manifest: {manifest_path}")

    # Counts by split
    items_by_split = defaultdict(list)
    for it in all_items:
        items_by_split[assignment[it["id"]]].append(it)

    counts = pretty_counts_by_split(items_by_split, num_classes)
    # Save counts as JSON for quick inspection
    with open(out_root / "class_counts_by_split.json", "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
    print(f"Wrote class count summary: {out_root/'class_counts_by_split.json'}")

    if args.copy:
        for split in ["train","val","test"]:
            (out_root / split / "images").mkdir(parents=True, exist_ok=True)
            (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

        for it in all_items:
            split = assignment[it["id"]]
            dst_img = out_root / split / "images" / it["img_path"].name
            dst_lbl = out_root / split / "labels" / it["lbl_path"].name
            shutil.copy2(it["img_path"], dst_img)
            if it["lbl_path"].exists():
                shutil.copy2(it["lbl_path"], dst_lbl)

        print(f"Copied images/labels into {out_root}.")

    print("Done.")

if __name__ == "__main__":
    main()