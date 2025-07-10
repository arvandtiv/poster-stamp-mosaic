import cv2
import os
import json
import numpy as np

# This script analyzes all extracted stamps to compute color and brightness statistics
# It processes stamps from all extracted_stamps_* directories and creates a comprehensive analysis

# Find all extracted_stamps_* folders in the current directory
# These folders contain the individual stamp images extracted from batch scans
batch_dirs = [d for d in os.listdir('.') if d.startswith('extracted_stamps_') and os.path.isdir(d)]
all_results = []

# Process each batch directory
for stamps_dir in batch_dirs:
    # Look for the positions.json file that contains metadata about stamp locations
    positions_file = os.path.join(stamps_dir, 'positions.json')
    if not os.path.exists(positions_file):
        continue
    
    # Load the position metadata for this batch
    with open(positions_file, 'r') as f:
        positions = json.load(f)
    
    # Analyze each stamp in this batch
    for entry in positions:
        # Construct the full path to the stamp image
        stamp_path = os.path.join(stamps_dir, entry['file'])
        
        # Load the stamp image using OpenCV
        img = cv2.imread(stamp_path)  # type: ignore
        if img is None:
            continue
        
        # Calculate average color in BGR format (OpenCV default)
        # This gives us the dominant color of each stamp
        avg_color = img.mean(axis=(0, 1))
        
        # Convert to grayscale and calculate average brightness
        # This helps identify light vs dark stamps
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # type: ignore
        avg_brightness = float(gray.mean())
        
        # Store analysis results for this stamp
        all_results.append({
            'row': entry['row'],           # Row position in original grid
            'col': entry['col'],           # Column position in original grid
            'file': entry['file'],         # Filename of the stamp image
            'batch': stamps_dir,           # Which batch this stamp came from
            'avg_color_bgr': avg_color.tolist(),  # Average color in BGR format
            'avg_brightness': avg_brightness      # Average brightness (0-255)
        })

# Save all analysis results to a JSON file
# This creates a comprehensive dataset of all stamp characteristics
with open('all_stamps_analysis.json', 'w') as f:
    json.dump(all_results, f, indent=2) 