#!/usr/bin/env python3
"""
Synthetic dataset generator for HVAC/BMS symbol detection (YOLO‑format).

Revision C (2025‑05‑26)
-----------------------
**Bug‑fix:** File names are now consistent.  The generator writes exactly one
image/label pair per index with **three‑digit zero‑padding**:
```
synthetic_000.jpg
synthetic_000.txt
synthetic_001.jpg
...
```
Remove any previously generated `synthetic_0.*` leftovers before re‑running.
All other behaviour (uniform icon size, placement logic) is unchanged.
"""

import cv2
import random
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Raw class dictionary (unchanged) – only used for name matching.
# ---------------------------------------------------------------------------

# (dictionary omitted here for brevity – unchanged from previous revision)

# <<< KEEP THE FULL dict_original_class_id AND OTHER SECTIONS UNCHANGED >>>
# To save space, only the main loop has been patched below.

# ---------------------------------------------------------------------------
# 8. Main entry point (patched naming)
# ---------------------------------------------------------------------------

def main():
    rng = random.Random()
    for i in range(NUM_IMAGES):
        img, lbl = generate_image(rng)

        stem = f"synthetic_{i:03d}"  # <<< three‑digit padding, single source of truth
        img_path = OUT_IMG_DIR / f"{stem}.jpg"
        lbl_path = OUT_LABEL_DIR / f"{stem}.txt"

        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(lbl_path, "w") as f:
            for c, cx, cy, bw, bh in lbl:
                f.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        if (i + 1) % 10 == 0 or i == 0:
            print(f"Generated {i + 1}/{NUM_IMAGES} → {img_path.name}")

    print("✅ Synthetic dataset generation complete – files are synthetic_000 .. synthetic_%03d" % (NUM_IMAGES-1))


if __name__ == "__main__":
    main()
