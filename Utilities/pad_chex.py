"""
pad_chex.py  –  Pre-process CheXpert_normal images into a flat 390x320 folder.

Rules (applied per image):
  height must be exactly 320 (all CheXpert-small images should satisfy this).

  needed = 390 - width
  needed == 0          →  copy as-is (already 390 wide).
  1 <= needed <= 10    →  zero-pad to reach 390: floor(needed/2) cols on the
                          left, ceil(needed/2) cols on the right.
                          This covers widths 380-389, using at most 5 cols per
                          side.  Odd differences get the extra column on the
                          right (e.g. width 389: 0 left, 1 right).
  needed > 10          →  skip (too narrow; would require > 5 cols per side).
  needed < 0           →  skip (wider than target).

Output flat folder:  <project_root>/chex_train/
Flat filename format: <patient>_<study>_<original_filename>.jpg
  e.g.  patient45768_study1_view1_frontal.jpg

Requirements:  pip install Pillow
"""

import os
import sys
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("ERROR: Pillow is not installed.  Run:  pip install Pillow")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths  (defaults; override with command-line args)
#   Usage:  python pad_chex.py [src_train_folder] [dst_flat_folder]
#   e.g.:   python pad_chex.py CheXpert_pacemaker/train chex_pacemaker
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR    = SCRIPT_DIR / (sys.argv[1] if len(sys.argv) > 1 else "CheXpert_normal/train")
DST_DIR    = SCRIPT_DIR / (sys.argv[2] if len(sys.argv) > 2 else "chex_train")

TARGET_W     = 390
TARGET_H     = 320
MAX_PAD_SIDE = 5      # maximum columns added to either side

SAVE_QUALITY = 95     # JPEG quality for re-saved images

# ---------------------------------------------------------------------------

def main():
    if not SRC_DIR.is_dir():
        print(f"ERROR: source folder not found:\n  {SRC_DIR}")
        sys.exit(1)

    DST_DIR.mkdir(exist_ok=True)
    print(f"Source : {SRC_DIR}")
    print(f"Dest   : {DST_DIR}")
    print(f"Target : {TARGET_W}x{TARGET_H}  (pad up to {MAX_PAD_SIDE} cols per side; widths {TARGET_W - 2*MAX_PAD_SIDE}-{TARGET_W} accepted)\n")

    jpg_files = sorted(SRC_DIR.rglob("*.jpg"))
    total = len(jpg_files)
    print(f"Found {total} .jpg files\n")

    copied  = 0   # already 390x320
    padded  = 0   # padded from 380 to 390
    skipped = 0   # wrong dimensions
    errors  = 0

    for i, src_path in enumerate(jpg_files, 1):
        if i % 2000 == 0:
            print(f"  [{i:>6}/{total}]  copied={copied}  padded={padded}  skipped={skipped}")

        try:
            img = Image.open(src_path)

            # Ensure single-channel grayscale
            if img.mode != 'L':
                img = img.convert('L')

            w, h = img.size   # PIL convention: (width, height)

            if h != TARGET_H:
                skipped += 1
                continue

            needed = TARGET_W - w

            if needed < 0 or needed > 2 * MAX_PAD_SIDE:
                # Too wide, or too narrow to fix within the per-side limit
                skipped += 1
                continue

            arr = np.array(img, dtype=np.uint8)

            if needed == 0:
                action = 'copy'
            else:
                pad_left  = needed // 2
                pad_right = needed - pad_left   # ceil; takes the odd col on right
                arr = np.pad(arr,
                             pad_width=((0, 0), (pad_left, pad_right)),
                             mode='constant',
                             constant_values=0)
                action = 'pad'

            # Build flat output filename
            # src_path relative to SRC_DIR looks like:
            #   patient45768/study1/view1_frontal.jpg
            parts = src_path.relative_to(SRC_DIR).parts
            flat_name = "_".join(parts)          # patient45768_study1_view1_frontal.jpg
            dst_path  = DST_DIR / flat_name

            # Save (always re-encode so the filename can differ from source)
            Image.fromarray(arr, mode='L').save(
                dst_path, format="JPEG", quality=SAVE_QUALITY, subsampling=0
            )

            if action == 'copy':
                copied += 1
            else:
                padded += 1

        except Exception as exc:
            print(f"  ERROR  {src_path.name}: {exc}")
            errors += 1

    print(f"\n{'='*50}")
    print(f"Finished processing {total} source files")
    print(f"  Copied  (already {TARGET_W}x{TARGET_H})         : {copied:>6}")
    print(f"  Padded  (widths {TARGET_W - 2*MAX_PAD_SIDE}-{TARGET_W - 1} -> {TARGET_W}) : {padded:>6}")
    print(f"  Skipped (wrong dimensions) : {skipped:>6}")
    print(f"  Errors                     : {errors:>6}")
    print(f"  Total written              : {copied + padded:>6}")
    print(f"\nOutput folder: {DST_DIR}")


if __name__ == "__main__":
    main()
