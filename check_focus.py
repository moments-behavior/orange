#!/usr/bin/env python3
"""
Focus quality checker for orange camera system (image file analyzer).

For pre-recording live camera checks, use the C++ binary instead:
    cd build && make focus_check
    ./check_focus_all.sh           # checks main machine + dosa-0 + dosa-1

This script is for analyzing already-saved image files (e.g. frames saved
with the orange "Take Picture" button, or frames extracted from recordings).

Usage:
    python3 check_focus.py /home/user/orange_data/pictures/
    python3 check_focus.py /path/to/image1.png /path/to/image2.jpg
    python3 check_focus.py --threshold 150 /path/to/images/
    python3 check_focus.py --save-dir /tmp/frames /path/to/images/  # also saves frames saved by focus_check --save-frames

The sharpness score is Laplacian variance — higher = sharper.
    < 50   : Very blurry, almost certainly out of focus
    50-100 : Borderline, check visually
    > 100  : Acceptably sharp (default threshold)
    > 300  : Very sharp
"""

import argparse
import sys
import os
import glob

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: OpenCV not found. Install with: pip3 install opencv-python")
    sys.exit(1)


def laplacian_variance(image_path):
    """Compute Laplacian variance sharpness score for an image."""
    img = cv2.imread(image_path)
    if img is None:
        return None, f"Could not read image: {image_path}"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score, None


def collect_images(paths):
    """Collect all image file paths from the given list of files/directories."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    result = []
    for p in paths:
        if os.path.isdir(p):
            for ext in image_extensions:
                result.extend(glob.glob(os.path.join(p, f"*{ext}")))
                result.extend(glob.glob(os.path.join(p, f"*{ext.upper()}")))
        elif os.path.isfile(p):
            result.append(p)
        else:
            print(f"WARNING: Path not found: {p}")
    return sorted(set(result))


def main():
    parser = argparse.ArgumentParser(
        description="Check camera focus sharpness before recording."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="Image file(s) or directory containing images",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Minimum acceptable sharpness score (default: 100)",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Print warnings but always exit with code 0",
    )
    args = parser.parse_args()

    images = collect_images(args.paths)
    if not images:
        print("ERROR: No image files found.")
        sys.exit(1)

    print(f"\n{'Image':<55} {'Score':>8}  {'Status'}")
    print("-" * 75)

    all_pass = True
    results = []
    for img_path in images:
        score, err = laplacian_variance(img_path)
        name = os.path.basename(img_path)
        if err:
            print(f"{name:<55} {'ERROR':>8}  {err}")
            all_pass = False
            results.append((name, None, False))
            continue

        passed = score >= args.threshold
        status = "OK" if passed else f"BLURRY (threshold: {args.threshold:.0f})"
        flag = "" if passed else "  <-- WARNING"
        print(f"{name:<55} {score:>8.1f}  {status}{flag}")
        if not passed:
            all_pass = False
        results.append((name, score, passed))

    print("-" * 75)

    failed = [(n, s) for n, s, p in results if not p]
    if failed:
        print(f"\n[FAIL] {len(failed)} camera(s) may not be in focus:")
        for name, score in failed:
            score_str = f"{score:.1f}" if score is not None else "ERROR"
            print(f"  - {name}  (score: {score_str})")
        print(
            "\nTips:"
            "\n  1. Check the 'focus' value in the camera's JSON config file"
            "\n     (e.g. config/<serial>.json). A value of 0 may mean focus"
            "\n     was never set — the camera stayed at factory default."
            "\n  2. Use the Focus slider in orange's GUI to find the sharpest"
            "\n     position, then save that value to the config JSON."
            "\n  3. In camera.cpp:update_focus_value(), if focus_value < focus_min"
            "\n     the set is silently skipped. Check the camera's focus_min."
        )
        if not args.warn_only:
            sys.exit(1)
    else:
        print(f"\n[PASS] All {len(results)} camera(s) appear to be in focus.")

    sys.exit(0)


if __name__ == "__main__":
    main()
