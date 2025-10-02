#!/usr/bin/env python3
"""
Auto-label YOLO OBB from videos using priors learned from hand-labeled CSVs.

What it does
------------
1) Learn per-class priors (area, aspect ratio, long-axis angle) from one or more CSVs that contain
   oriented 4-corner labels (same format you use to export YOLO OBB). CSVs must include columns:
   frame, <ignored>, class_id, x1,y1,x2,y2,x3,y3,x4,y4
2) Discover videos (multiple date-tag roots or explicit paths).
3) For each frame (optionally stride): run MATLAB-style motion detector to get candidate OBBs.
4) For each candidate, compute (area, aspect, angle) and classify against learned priors.
   - Only keep detections that match one of the known classes within robust gates (IQR-based).
   - Write YOLO OBB labels with the chosen class id.
5) Split into train/val/test and export full YOLO layout (+ test overlays).

Assumptions
-----------
- Three classes exist in your CSV(s). Their numeric IDs (e.g., 0,1,2) are preserved in output.
- If you supply a class names file, it is only used for data.yaml (does not affect numeric mapping).

Usage
-----
python autolabel_yolo_obb_with_priors.py \
  --label_csvs /data/labels/Cam2005325_obb.csv \
  -v /data/videos/2025_09_20_12_00_00 -v /data/videos/2025_09_21_12_00_00 \
  -o /data/yolo_obb_auto_priors \
  --image_size 640 --bg_mode first --threshold 36 --frame_stride 1 \
  --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1 \
  --viz_test true --class_names sphere,vertical_cylinder,horizontal_cylinder

"""

import argparse
import os
import re
import cv2
import yaml
import math
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# ----------------------------
# Date tag helpers
# ----------------------------
_TS_RE = re.compile(r'^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$')

def find_timestamp_in_path(path: str) -> Optional[str]:
    cur = os.path.abspath(path)
    if os.path.isfile(cur):
        cur = os.path.dirname(cur)
    while True:
        base = os.path.basename(cur)
        if _TS_RE.match(base):
            return base
        nxt = os.path.dirname(cur)
        if nxt == cur:
            return None
        cur = nxt

# ----------------------------
# Video discovery
# ----------------------------
def parse_video_files(video_roots: List[str], explicit_video_files: List[str]) -> Dict[Tuple[str, Optional[str]], str]:
    video_exts = {'.mp4','.avi','.mov','.mkv'}
    mapping: Dict[Tuple[str, Optional[str]], str] = {}
    def _add(p: str):
        if not os.path.isfile(p): return
        ext = os.path.splitext(p)[1].lower()
        if ext not in video_exts: return
        cam = os.path.splitext(os.path.basename(p))[0]
        tag = find_timestamp_in_path(p)
        mapping[(cam, tag)] = p

    for root in video_roots:
        if not os.path.isdir(root): continue
        for dirpath,_,files in os.walk(root):
            for fname in files:
                _add(os.path.join(dirpath, fname))

    for v in explicit_video_files:
        for part in [pp.strip() for pp in v.split(',') if pp.strip()]:
            _add(part)

    return mapping

# ----------------------------
# Morphology & geometry
# ----------------------------
def disk_kernel(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    y,x = np.ogrid[-r:r+1, -r:r+1]
    mask = x*x + y*y <= r*r
    k = np.zeros((2*r+1, 2*r+1), dtype=np.uint8)
    k[mask] = 1
    return k

def order_corners_clockwise_start_tl(pts: np.ndarray) -> np.ndarray:
    P = pts.astype(np.float32)
    c = P.mean(axis=0)
    ang = np.arctan2(P[:,1]-c[1], P[:,0]-c[0])  # [-pi,pi]
    idx = np.argsort(ang)  # CCW
    P = P[idx]
    start = np.lexsort((P[:,0], P[:,1]))[0]     # min y then min x
    P = np.roll(P, -start, axis=0)
    return P

def long_axis_angle_deg(pts: np.ndarray) -> float:
    """Return angle (0..180) of the long rectangle axis from OBB corners (order-agnostic)."""
    P = order_corners_clockwise_start_tl(pts)
    # choose the longer of edges (p0->p1) vs (p1->p2)
    e01 = P[1] - P[0]
    e12 = P[2] - P[1]
    len01 = np.linalg.norm(e01)
    len12 = np.linalg.norm(e12)
    v = e01 if len01 >= len12 else e12
    ang = (math.degrees(math.atan2(v[1], v[0])) + 180.0) % 180.0
    return ang

def rect_wh_from_pts(pts: np.ndarray) -> Tuple[float,float]:
    P = order_corners_clockwise_start_tl(pts)
    e01 = np.linalg.norm(P[1] - P[0])
    e12 = np.linalg.norm(P[2] - P[1])
    w, h = (e01, e12) if e01 >= e12 else (e12, e01)
    return w, h

# ----------------------------
# Background & detection (MATLAB-like)
# ----------------------------
def build_background(cap: cv2.VideoCapture, mode: str = 'first', frames: int = 30) -> Optional[np.ndarray]:
    if mode == 'first':
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, fr = cap.read()
        return fr if ok else None
    elif mode == 'median':
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        imgs = []
        for _ in range(max(1, frames)):
            ok, fr = cap.read()
            if not ok: break
            imgs.append(fr.astype(np.float32))
        if not imgs: return None
        med = np.median(np.stack(imgs,0), axis=0).astype(np.uint8)
        return med
    else:
        raise ValueError(f"Unknown bg mode: {mode}")

def detect_candidates(frame_bgr: np.ndarray, bg_bgr: np.ndarray, thresh: int = 36) -> List[np.ndarray]:
    diff = cv2.subtract(frame_bgr, bg_bgr)    # BGR subtraction
    chan = diff[:,:,1]                        # use green channel like MATLAB (:,:,1)
    _, mask = cv2.threshold(chan, int(thresh), 255, cv2.THRESH_BINARY)
    mask = cv2.erode(mask, disk_kernel(2), iterations=1)
    mask = cv2.dilate(mask, disk_kernel(5), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obb_list: List[np.ndarray] = []
    for cnt in contours:
        if len(cnt) < 5: continue
        rect = cv2.minAreaRect(cnt)
        pts  = cv2.boxPoints(rect)           # 4x2 float
        pts  = order_corners_clockwise_start_tl(pts)
        obb_list.append(pts.astype(np.float32))
    return obb_list

# ----------------------------
# Priors from CSV(s)
# ----------------------------
def parse_labels_csv(csv_path: str) -> List[Tuple[int, np.ndarray]]:
    """Return list of (class_id, 4x2 pts) from a CSV with columns frame,?,class_id,x1,y1,...,x4,y4"""
    out = []
    if not os.path.isfile(csv_path):
        return out
    with open(csv_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return out
    # skip possible title line
    if not lines[0].lower().startswith('frame'):
        lines = lines[1:] if len(lines) > 1 else []
    if not lines:
        return out
    for row in lines[1:]:
        parts = row.split(',')
        if len(parts) < 11: continue
        try:
            cls = int(float(parts[2]))
            coords = [float(x) for x in parts[3:11]]
            pts = np.array(coords, dtype=np.float32).reshape(4,2)
            out.append((cls, pts))
        except:
            continue
    return out

def iqr_bounds(vals: np.ndarray, k: float = 2.0) -> Tuple[float,float,float,float]:
    """Return (median, iqr, lo, hi) with lo/hi = median ± k*IQR (clamped >=0)."""
    vals = np.asarray(vals).astype(np.float32)
    if vals.size == 0:
        return 0.0, 1.0, -np.inf, np.inf
    q1 = np.percentile(vals, 25)
    q3 = np.percentile(vals, 75)
    med = np.median(vals)
    iqr = max(1e-6, q3 - q1)
    lo = max(0.0, med - k * iqr)
    hi = med + k * iqr
    return float(med), float(iqr), float(lo), float(hi)

def angular_stats(angles_deg: np.ndarray, k: float = 2.0) -> Tuple[float,float,float,float]:
    """Compute median and IQR on wrapped angles in [0,180)."""
    a = np.mod(angles_deg, 180.0)
    med = float(np.median(a))
    # project to nearest of med to compute IQR approximately
    diffs = np.abs(((a - med + 90) % 180) - 90)
    q1 = np.percentile(diffs, 25)
    q3 = np.percentile(diffs, 75)
    iqr = max(1e-6, q3 - q1)
    lo = 0.0
    hi = 180.0
    # store med and iqr; during gating we compare circular diffs to k*iqr
    return med, iqr, lo, hi

def learn_priors_from_csvs(csv_paths: List[str]) -> Dict[int, Dict[str, Any]]:
    """
    For each class id present in CSVs, learn priors:
     - area = w*h
     - aspect = (long/short)
     - angle = long-axis angle in [0,180)
    Returns {class_id: {'area':(med,iqr,lo,hi), 'aspect':(...), 'angle':(med,iqr,0,180)}}
    """
    per_class_samples: Dict[int, Dict[str, List[float]]] = {}
    for p in csv_paths:
        for (cls, pts) in parse_labels_csv(p):
            w, h = rect_wh_from_pts(pts)
            area = w * h
            aspect = (w / max(1e-6, h))
            ang = long_axis_angle_deg(pts)
            d = per_class_samples.setdefault(cls, {'area':[], 'aspect':[], 'angle':[]})
            d['area'].append(area)
            d['aspect'].append(aspect)
            d['angle'].append(ang)

    priors: Dict[int, Dict[str, Any]] = {}
    for cls, d in per_class_samples.items():
        area_med, area_iqr, area_lo, area_hi = iqr_bounds(np.array(d['area']), k=2.0)
        asp_med, asp_iqr, asp_lo, asp_hi   = iqr_bounds(np.array(d['aspect']), k=2.0)
        ang_med, ang_iqr, _, _             = angular_stats(np.array(d['angle']), k=2.0)
        priors[cls] = {
            'area':  (area_med, area_iqr, area_lo, area_hi),
            'aspect':(asp_med,  asp_iqr,  asp_lo,  asp_hi),
            'angle': (ang_med,  ang_iqr,  0.0,     180.0)
        }
    return priors

def circular_diff(a: float, b: float) -> float:
    """Smallest absolute difference on circle [0,180)."""
    d = abs((a - b + 90.0) % 180.0 - 90.0)
    return d

def classify_with_priors(pts: np.ndarray, priors: Dict[int, Dict[str, Any]], gates: Dict[str, float]) -> Optional[int]:
    """Return class id with best (lowest) normalized score if within gates; else None."""
    if not priors: return None
    w, h = rect_wh_from_pts(pts)
    area = w * h
    aspect = w / max(1e-6, h)
    ang = long_axis_angle_deg(pts)

    best_cls = None
    best_score = 1e9

    for cls, P in priors.items():
        area_med, area_iqr, area_lo, area_hi = P['area']
        asp_med, asp_iqr, asp_lo, asp_hi     = P['aspect']
        ang_med, ang_iqr, _, _               = P['angle']

        # gate checks
        if not (area_lo * gates['area_lo'] <= area <= area_hi * gates['area_hi']):
            continue
        if not (asp_lo  * gates['asp_lo']  <= aspect <= asp_hi   * gates['asp_hi']):
            continue
        # angle gate by circular distance vs k*IQR (scaled)
        ang_dev = circular_diff(ang, ang_med)
        if ang_dev > gates['ang_k'] * max(ang_iqr, 5.0):  # at least 5deg tolerance if IQR tiny
            continue

        # score = sum of normalized deviations
        s_area = abs(area - area_med) / max(area_iqr, 1e-6)
        s_asp  = abs(aspect - asp_med) / max(asp_iqr, 1e-6)
        s_ang  = ang_dev / max(ang_iqr, 5.0)
        score  = s_area + s_asp + s_ang

        if score < best_score:
            best_score = score
            best_cls = cls

    return best_cls

# ----------------------------
# Letterbox & normalize
# ----------------------------
def resize_image_for_yolo(image, target_size=640):
    h, w = image.shape[:2]
    scale = target_size / max(h, w) if max(h, w) > 0 else 1.0
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas, scale, x_offset, y_offset

def adjust_points_for_resize(points_xy: np.ndarray, scale: float, x_offset: int, y_offset: int) -> np.ndarray:
    pts = points_xy.copy().astype(np.float32)
    pts[:,0] = pts[:,0] * scale + x_offset
    pts[:,1] = pts[:,1] * scale + y_offset
    return pts

def normalize_points(points_xy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    pts = points_xy.copy().astype(np.float32)
    pts[:,0] = pts[:,0] / img_w
    pts[:,1] = pts[:,1] / img_h
    return pts

# ----------------------------
# Dataset helpers
# ----------------------------
def create_dataset_structure(output_dir: str):
    for split in ['train','val','test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

def split_keys(keys: List[str], train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = random.Random(seed)
    arr = list(keys)
    rng.shuffle(arr)
    n = len(arr)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train = arr[:n_train]
    val   = arr[n_train:n_train+n_val]
    test  = arr[n_train+n_val:]
    return {'train': train, 'val': val, 'test': test}

def write_data_yaml(output_dir: str, names: List[str]):
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(names),
        'names': names
    }
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

# ----------------------------
# Visualization (test split)
# ----------------------------
def visualize_test_split(output_dir: str, image_size: int, line_thickness: int = 2):
    labels_dir = os.path.join(output_dir, 'train', 'labels')
    images_dir = os.path.join(output_dir, 'train', 'images')
    vis_dir    = os.path.join(output_dir, 'train', 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    if not os.path.isdir(labels_dir):
        print(f"[viz] No labels at {labels_dir}")
        return

    for lbl_name in sorted(os.listdir(labels_dir)):
        if not lbl_name.endswith('.txt'): continue
        base = os.path.splitext(lbl_name)[0]
        img_path = os.path.join(images_dir, base + '.jpg')
        if not os.path.isfile(img_path):
            print(f"[viz] Missing image for {lbl_name}")
            continue
        img = cv2.imread(img_path)
        with open(os.path.join(labels_dir, lbl_name),'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) < 9: continue
            cls = int(float(parts[0]))
            coords = [float(x) for x in parts[1:9]]
            pts = []
            for i in range(0,8,2):
                x = int(round(coords[i]   * image_size))
                y = int(round(coords[i+1] * image_size))
                pts.append([x,y])
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [pts], True, (0,255,0), line_thickness)
            cv2.putText(img, str(cls), (pts[0][0], pts[0][1]-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(vis_dir, base + '.jpg'), img)
    print(f"[viz] Wrote overlays to {vis_dir}")

# ----------------------------
# End-to-end
# ----------------------------
def autolabel_with_priors(label_csvs: List[str], video_map: Dict[Tuple[str, Optional[str]], str], output_dir: str,
                          image_size: int = 640, bg_mode: str = 'first', bg_frames: int = 30,
                          thresh: int = 36, frame_stride: int = 1, keep_negatives: bool = True,
                          train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1,
                          seed: int = 42, viz_test: bool = True, viz_line_thickness: int = 2,
                          class_names: Optional[List[str]] = None):
    os.makedirs(output_dir, exist_ok=True)
    create_dataset_structure(output_dir)

    # Learn priors
    priors = learn_priors_from_csvs(label_csvs)
    if not priors:
        raise RuntimeError("No priors learned. Check --label_csvs input.")
    print("[priors] Learned for classes:", sorted(priors.keys()))

    # Gates for robustness (multipliers around IQR bounds)
    gates = {
        'area_lo': 0.9,   # allow slightly smaller than lo
        'area_hi': 1.1,   # allow slightly larger than hi
        'asp_lo':  0.9,
        'asp_hi':  1.1,
        'ang_k':   2.0    # allow up to 2*IQR angular deviation (>=5deg floor applied)
    }

    # Pass 1: collect keys + detections
    detections_by_key: Dict[str, List[Tuple[int, np.ndarray]]] = {}
    meta_by_key: Dict[str, Tuple[str, Optional[str], int]] = {}

    for (cam, tag), vpath in video_map.items():
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"[warn] Cannot open {vpath}")
            continue
        bg = build_background(cap, mode=bg_mode, frames=bg_frames)
        if bg is None:
            print(f"[warn] Cannot build background for {vpath}")
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            candidates = detect_candidates(fr, bg, thresh=thresh)
            labeled_obbs: List[Tuple[int, np.ndarray]] = []
            for pts in candidates:
                cls = classify_with_priors(pts, priors, gates)
                if cls is not None:
                    labeled_obbs.append((cls, pts))

            key = f"{cam}_{tag or 'NA'}_{frame_idx}"
            if labeled_obbs or keep_negatives:
                detections_by_key[key] = labeled_obbs
                meta_by_key[key] = (vpath, tag, frame_idx)

            frame_idx += 1

        cap.release()

    if not detections_by_key:
        raise RuntimeError("No frames collected after classification; detector too strict? Try relaxing gates or threshold.")

    # Split
    splits = split_keys(list(detections_by_key.keys()), train_ratio, val_ratio, test_ratio, seed=seed)
    print(f"Split sizes: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Pass 2: write images & labels
    def ensure_dirs(split: str):
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    for split in ['train','val','test']:
        ensure_dirs(split)
        for key in splits[split]:
            vpath, tag, frame_idx = meta_by_key[key]
            cap = cv2.VideoCapture(vpath)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, fr = cap.read()
            cap.release()
            if not ok:
                print(f"[warn] Failed to re-read {vpath} frame {frame_idx}")
                continue

            letterboxed, scale, xoff, yoff = resize_image_for_yolo(fr, image_size)
            labeled = detections_by_key[key]
            # transform & normalize
            yolo_lines = []
            for (cls, pts) in labeled:
                pts_resized = adjust_points_for_resize(pts, scale, xoff, yoff)
                pts_norm    = normalize_points(pts_resized, image_size, image_size)
                yolo_lines.append((cls, pts_norm))

            img_name = f"{key}.jpg"
            lbl_name = f"{key}.txt"
            cv2.imwrite(os.path.join(output_dir, split, 'images', img_name), letterboxed)

            with open(os.path.join(output_dir, split, 'labels', lbl_name), 'w') as f:
                for (cls, pts) in yolo_lines:
                    flat = pts.reshape(-1).tolist()
                    f.write(str(cls) + " " + " ".join(f"{v:.6f}" for v in flat) + "\n")

    # data.yaml
    if class_names is None:
        # default names by sorted class ids so names length == max_id+1
        max_cls = max(priors.keys())
        names = ['class_'+str(i) for i in range(max_cls+1)]
    else:
        names = class_names
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump({
            'path': os.path.abspath(output_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(names),
            'names': names
        }, f, default_flow_style=False)

    # Visualize test split
    if viz_test:
        visualize_test_split(output_dir, image_size, line_thickness=viz_line_thickness)

# ----------------------------
# CLI
# ----------------------------
def str2bool(v):
    if isinstance(v, bool): return v
    s = str(v).lower().strip()
    if s in ('true','1','yes','y','on'): return True
    if s in ('false','0','no','n','off'): return False
    raise argparse.ArgumentTypeError(f"Boolean expected: {v}")

def parse_list(values: List[str]) -> List[str]:
    out = []
    for v in values or []:
        out.extend([p.strip() for p in v.split(',') if p.strip()])
    return out

def main():
    ap = argparse.ArgumentParser(description="Auto-label videos -> YOLO OBB using priors learned from hand labels")
    ap.add_argument('--label_csvs', action='append', default=[],
                    help='Hand-labeled CSV files (repeatable or comma-separated). Used to learn size/aspect/angle priors.')
    ap.add_argument('-v','--video_dir', action='append', default=[],
                    help='Video root dir (repeatable). Recursively indexes videos and infers (camera_id, date_tag).')
    ap.add_argument('--video_files', action='append', default=[],
                    help='Explicit video file paths (repeatable or comma-separated).')
    ap.add_argument('-o','--output_dir', required=True, type=str, help='Output dataset root')
    ap.add_argument('--image_size', type=int, default=640, help='Square export size (default=640)')
    ap.add_argument('--bg_mode', type=str, default='first', choices=['first','median'], help='Background mode')
    ap.add_argument('--bg_frames', type=int, default=30, help='Frames for median background')
    ap.add_argument('--threshold', type=int, default=36, help='Binary threshold after subtraction')
    ap.add_argument('--frame_stride', type=int, default=1, help='Process every K-th frame')
    ap.add_argument('--keep_negatives', type=str2bool, default=True, help='Write empty labels for frames with no accepted detections')
    ap.add_argument('--train_ratio', type=float, default=0.7)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--test_ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--viz_test', type=str2bool, default=True, help='Render OBB overlays for test split')
    ap.add_argument('--viz_line_thickness', type=int, default=2)
    ap.add_argument('--class_names', type=str, default=None,
                    help='Comma-separated class names to write into data.yaml (e.g., "sphere,vertical_cylinder,horizontal_cylinder")')
    args = ap.parse_args()

    label_csvs = parse_list(args.label_csvs)
    if not label_csvs:
        raise SystemExit("Please provide at least one --label_csvs path to learn priors.")

    video_map = parse_video_files(parse_list(args.video_dir), parse_list(args.video_files))
    if not video_map:
        raise SystemExit("No videos found. Use -v roots and/or --video_files.")

    class_names = [s.strip() for s in args.class_names.split(',')] if args.class_names else None

    autolabel_with_priors(
        label_csvs=label_csvs,
        video_map=video_map,
        output_dir=args.output_dir,
        image_size=args.image_size,
        bg_mode=args.bg_mode,
        bg_frames=args.bg_frames,
        thresh=args.threshold,
        frame_stride=args.frame_stride,
        keep_negatives=args.keep_negatives,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        viz_test=args.viz_test,
        viz_line_thickness=args.viz_line_thickness,
        class_names=class_names
    )

if __name__ == "__main__":
    main()
