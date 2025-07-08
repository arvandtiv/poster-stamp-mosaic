import cv2
import os
import json
import numpy as np

# Find all extracted_stamps_* folders
batch_dirs = [d for d in os.listdir('.') if d.startswith('extracted_stamps_') and os.path.isdir(d)]
all_results = []

for stamps_dir in batch_dirs:
    positions_file = os.path.join(stamps_dir, 'positions.json')
    if not os.path.exists(positions_file):
        continue
    with open(positions_file, 'r') as f:
        positions = json.load(f)
    for entry in positions:
        stamp_path = os.path.join(stamps_dir, entry['file'])
        img = cv2.imread(stamp_path)
        if img is None:
            continue
        avg_color = img.mean(axis=(0, 1))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg_brightness = float(gray.mean())
        all_results.append({
            'row': entry['row'],
            'col': entry['col'],
            'file': entry['file'],
            'batch': stamps_dir,
            'avg_color_bgr': avg_color.tolist(),
            'avg_brightness': avg_brightness
        })

with open('all_stamps_analysis.json', 'w') as f:
    json.dump(all_results, f, indent=2) 