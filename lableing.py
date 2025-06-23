#!/usr/bin/env python3
"""
Synthetic dataset generator for HVAC/BMS symbol detection (YOLO‑format).

Revision B (2025‑06‑23)
-----------------------
* **Removed automatic symbol scaling** – components now retain their
  original pixel dimensions.
* Minor cleanup of now‑unused constants.

Edit constants in *section 5* as needed.
"""

import cv2
import random
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Raw class dictionary (unchanged) – used only for name matching.
# ---------------------------------------------------------------------------

dict_original_class_id = {
    'shut-off_valve_on': 0,
    'shut-off_valve_off': 1,
    'pump_on_up': 2,
    'pump_on_right': 3,
    'pump_on_left': 4,
    'pump_on_down': 5,
    'pump_off_up': 6,
    'pump_off_right': 7,
    'pump_off_left': 8,
    'pump_off_down': 9,
    'metering_device': 10,
    'frequency_inverter_on': 11,
    'frequency_inverter_off': 12,
    'digital_volume_sensor_on': 13,
    'digital_volume_sensor_off': 14,
    'digital_temperature_sensor': 15,
    'digital_relative_humidity_sensor': 16,
    'digital_differential_pressure_sensor': 17,
    'digital_absolute_humidity_sensor': 18,
    'differential_pressure_sensor': 19,
    'analog_pressure_sensor': 20,
    'analogue_relative_humidity_sensor': 21,
    '3-way-control_valve': 22,
    '2-way-control_valve': 23,
    'chiller': 24,
    'chiller_off': 25,
    'chiller_on': 26,
    'combined_coarse_fine_filter_right': 27,
    'constant_volume_flow_controller_left': 28,
    'constant_volume_flow_controller_right': 29,
    'consumer': 30,
    'cooling_coil': 31,
    'damper_off': 32,
    'discharge_well': 33,
    'electical_heater': 34,
    'electrical_air_damper_on': 35,
    'electrical_hot_water_storage_tank': 36,
    'exhaust_air_combined_fine_coarse_filter_left': 37,
    'fan_on_left': 38,
    'fan_on_left_1': 39,
    'fan_on_right': 40,
    'fan_on_right_1': 41,
    'fan_on_right_2': 42,
    'fan_right': 43,
    'filter_left': 44,
    'filter_right': 45,
    'fine_filter': 46,
    'heat_exchanger': 47,
    'heat_exchanger_ventilation_system': 48,
    'heating_coil': 49,
    'inlet_vane_controlled_fan_off_left': 50,
    'inlet_vane_controlled_fan_off_right': 51,
    'inlet_vane_controlled_fan_on_left_1': 52,
    'inlet_vane_controlled_fan_on_right_1': 53,
    'pressurization_system': 54,
    'rooftop_chiller_unit': 55,
    'room_switch_off': 56,
    'room_switch_on': 57,
    'rotary_heat_exchanger': 58,
    'steam_humidifer': 59,
    'steam_humidifier_off': 60,
    'steam_humidifier_on': 61,
    'variable_volume_flow_controller_left': 62,
    'variable_volume_flow_controller_right': 63,
    'vertical_heat_exchanger_heating_cooling': 64,
    'water_storage_tank': 65,
    'water_storage_tank_1': 66
}

# ---------------------------------------------------------------------------
# 2. Raw → merged name mapping
# ---------------------------------------------------------------------------

class_group_mapping = {
    'pump_on_up': 'pump_on',
    'pump_on_right': 'pump_on',
    'pump_on_left': 'pump_on',
    'pump_on_down': 'pump_on',
    'pump_off_up': 'pump_off',
    'pump_off_right': 'pump_off',
    'pump_off_left': 'pump_off',
    'pump_off_down': 'pump_off',

    'fan_on_left': 'fan_on',
    'fan_on_left_1': 'fan_on',
    'fan_on_right': 'fan_on',
    'fan_on_right_1': 'fan_on',
    'fan_on_right_2': 'fan_on',
    'fan_right': 'fan_on',

    'filter_left': 'filter',
    'filter_right': 'filter',
    'combined_coarse_fine_filter_right': 'combined_coarse_fine_filter',
    'exhaust_air_combined_fine_coarse_filter_left': 'combined_coarse_fine_filter',

    'constant_volume_flow_controller_left': 'constant_volume_flow_controller',
    'constant_volume_flow_controller_right': 'constant_volume_flow_controller',
    'variable_volume_flow_controller_left': 'variable_volume_flow_controller',
    'variable_volume_flow_controller_right': 'variable_volume_flow_controller',

    'inlet_vane_controlled_fan_off_left': 'inlet_vane_controlled_fan_off',
    'inlet_vane_controlled_fan_off_right': 'inlet_vane_controlled_fan_off',
    'inlet_vane_controlled_fan_on_left_1': 'inlet_vane_controlled_fan_on',
    'inlet_vane_controlled_fan_on_right_1': 'inlet_vane_controlled_fan_on',

    'water_storage_tank_1': 'water_storage_tank'
}

# ---------------------------------------------------------------------------
# 3. Final class list and IDs (alphabetic)
# ---------------------------------------------------------------------------

general_names = {class_group_mapping.get(n, n) for n in dict_original_class_id}
final_class_names = sorted(general_names)
final_class_ids = {name: idx for idx, name in enumerate(final_class_names)}

# ---------------------------------------------------------------------------
# 4. Paths & global constants
# ---------------------------------------------------------------------------

GROUND_TRUTH_DIR = Path("ground_truth_dilt_labelled")
CONTEXT_DIR = Path("context_images")
OUT_IMG_DIR = Path("synthetic_dataset")
OUT_LABEL_DIR = Path("synthetic_labels")
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

NUM_IMAGES = 100
IMG_W, IMG_H = 1700, 800
BACKGROUND_RGB = (230, 178, 172)
MAX_PLACEMENT_TRIES = 300

# ---------------------------------------------------------------------------
# 5. Helper functions
# ---------------------------------------------------------------------------

def load_images(folder: Path):
    """Return list[(file_name, img_BGRorBGRA)]."""
    images = []
    for p in folder.glob("*.*"):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append((p.name, img))
    return images

def is_overlapping(x1, y1, w1, h1, x2, y2, w2, h2):
    return not (x1 + w1 <= x2 or x1 >= x2 + w2 or y1 + h1 <= y2 or y1 >= y2 + h2)

def generalise(raw_name: str) -> str:
    return class_group_mapping.get(raw_name, raw_name)

def class_id_from_file(fname: str) -> int:
    raw = fname.split('.')[0]
    for k in dict_original_class_id:
        if k in raw:
            return final_class_ids[generalise(k)]
    raise KeyError(fname)

def place_alpha(dst, src, tl):
    x, y = tl
    h, w = src.shape[:2]
    if src.shape[2] == 4:
        alpha = src[:, :, 3] / 255.0
        for c in range(3):
            dst[y:y+h, x:x+w, c] = (1-alpha) * dst[y:y+h, x:x+w, c] + alpha * src[:, :, c]
    else:
        dst[y:y+h, x:x+w] = src[:, :, :3]

# ---------------------------------------------------------------------------
# 6. Load resources (no scaling – symbols keep original size)
# ---------------------------------------------------------------------------

print("Loading symbols …")
SYMBOLS = load_images(GROUND_TRUTH_DIR)
if not SYMBOLS:
    raise SystemExit("No symbols found – check GROUND_TRUTH_DIR path")

# sort by area descending so big ones placed first
SYMBOLS.sort(key=lambda it: it[1].shape[0] * it[1].shape[1], reverse=True)

print(f"  {len(SYMBOLS)} symbols loaded")

print("Loading context images …")
CONTEXTS = [img for _, img in load_images(CONTEXT_DIR)]
print(f"  {len(CONTEXTS)} context images loaded")

# ---------------------------------------------------------------------------
# 7. Synthetic generator
# ---------------------------------------------------------------------------

def generate_image(rng: random.Random):
    canvas = np.full((IMG_H, IMG_W, 3), BACKGROUND_RGB, np.uint8)
    labels, occupied = [], []

    for fname, img in SYMBOLS:
        h, w = img.shape[:2]
        maxx, maxy = IMG_W - w, IMG_H - h
        placed = False
        for _ in range(MAX_PLACEMENT_TRIES):
            x, y = rng.randint(0, maxx), rng.randint(0, maxy)
            if any(is_overlapping(x, y, w, h, ox, oy, ow, oh) for ox, oy, ow, oh in occupied):
                continue
            place_alpha(canvas, img, (x, y))
            cx, cy = (x + w / 2) / IMG_W, (y + h / 2) / IMG_H
            labels.append((class_id_from_file(fname), cx, cy, w / IMG_W, h / IMG_H))
            occupied.append((x, y, w, h))
            placed = True
            break
        if not placed:
            print(f"⚠️ Could not place {fname}")

    # optional context clutter
    for ctx in CONTEXTS:
        h, w = ctx.shape[:2]
        maxx, maxy = IMG_W - w, IMG_H - h
        for _ in range(50):
            x, y = rng.randint(0, maxx), rng.randint(0, maxy)
            if any(is_overlapping(x, y, w, h, ox, oy, ow, oh) for ox, oy, ow, oh in occupied):
                continue
            place_alpha(canvas, ctx, (x, y))
            occupied.append((x, y, w, h))
            break

    return canvas, labels

# ---------------------------------------------------------------------------
# 8. Main entry point
# ---------------------------------------------------------------------------

def main():
    rng = random.Random()
    for i in range(NUM_IMAGES):
        img, lbl = generate_image(rng)
        img_path = OUT_IMG_DIR / f"synthetic_{i:04d}.jpg"
        lbl_path = OUT_LABEL_DIR / f"synthetic_{i:04d}.txt"
        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(lbl_path, "w") as f:
            for c, cx, cy, bw, bh in lbl:
                f.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Generated {i + 1}/{NUM_IMAGES} → {img_path.name}")
    print("✅ Synthetic dataset generation complete")

if __name__ == "__main__":
    main()
