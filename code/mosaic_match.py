import cv2
import numpy as np
import json
import os

# MOSAIC MATCHING SCRIPT
# This script creates a mosaic by matching individual stamps to target image cells
# It uses color and brightness analysis to find the best stamp for each position

# Configuration parameters
analysis_file = 'all_stamps_analysis.json'  # File containing stamp analysis data
target_image_file = 'target_image_100_100.jpg'  # Target image to recreate as mosaic
output_mosaic = 'mosaic_output.jpg'  # Output mosaic image
output_blueprint = 'mosaic_blueprint.json'  # Blueprint file with placement instructions

# Load batch metadata to get physical stamp dimensions
# This is used to calculate the final artwork size
with open('code/batch_metadata.json', 'r') as f:
    batch_metadata = json.load(f)
stamp_width_cm = batch_metadata['batches'][0]['stamp_width_cm']   # Physical width of stamps
stamp_height_cm = batch_metadata['batches'][0]['stamp_height_cm']  # Physical height of stamps

def load_stamps():
    """Load stamp analysis data from JSON file"""
    with open(analysis_file, 'r') as f:
        return json.load(f)

def load_and_resize_target(rows, cols):
    """Load and resize target image to match mosaic grid dimensions"""
    img = cv2.imread(target_image_file)  # type: ignore
    if img is None:
        raise FileNotFoundError(f"Target image not found: {target_image_file}")
    # Resize target to match the mosaic grid size
    return cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA)  # type: ignore

def compute_distance(stamp_color, stamp_brightness, target_color, target_brightness):
    """
    Compute similarity distance between a stamp and target cell
    Combines color distance (Euclidean) and brightness difference
    Lower distance = better match
    """
    color_dist = np.linalg.norm(np.array(stamp_color) - np.array(target_color))
    brightness_dist = abs(stamp_brightness - target_brightness)
    return color_dist + brightness_dist  # Simple sum, can be weighted

def get_rotated_stats(img):
    """
    Generate rotated versions of an image and compute their color/brightness stats
    Returns list of dictionaries with rotated images and their characteristics
    """
    stats = []
    for k, angle in enumerate([0, 90, 180, 270]):
        # Rotate image by 90 degrees k times
        rotated = np.ascontiguousarray(np.rot90(img, k=k))
        avg_color = rotated.mean(axis=(0, 1))
        avg_brightness = float(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY).mean())  # type: ignore
        stats.append({'img': rotated, 'angle': angle, 'avg_color': avg_color, 'avg_brightness': avg_brightness})
    return stats

def main():
    """Main mosaic creation algorithm"""
    # Load all stamp analysis data
    stamps = load_stamps()
    
    # Determine mosaic grid size based on stamp positions
    rows = max(s['row'] for s in stamps)
    cols = max(s['col'] for s in stamps)
    
    # Load and resize target image to match grid
    target = load_and_resize_target(rows, cols)
    
    # Initialize tracking variables
    blueprint = []  # Will store placement instructions
    used = set()    # Track which stamps have been used (no duplicates)

    # Determine cell size for the mosaic
    # Use the median size of all stamps for consistency
    stamp_shapes = []
    for stamp in stamps:
        stamp_img = cv2.imread(os.path.join(stamp['batch'], stamp['file']))  # type: ignore
        if stamp_img is not None:
            stamp_shapes.append(stamp_img.shape[:2])
    
    if not stamp_shapes:
        raise ValueError("No valid stamp images found.")
    
    # Calculate median dimensions for consistent cell size
    median_h = int(np.median([s[0] for s in stamp_shapes]))
    median_w = int(np.median([s[1] for s in stamp_shapes]))

    cell_h, cell_w = median_h, median_w
    
    # Create blank mosaic canvas
    mosaic = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    # Process each cell in the mosaic grid
    for r in range(rows):
        for c in range(cols):
            # Get target cell color and brightness
            cell = target[r, c]
            cell_brightness = float(np.mean(cell))
            
            # Find best matching stamp for this cell
            best_stamp = None
            best_dist = float('inf')
            best_rotation = 0
            best_img = None
            
            # Try each available stamp
            for stamp in stamps:
                # Skip if stamp already used (no duplicates)
                if (stamp['batch'], stamp['file']) in used:
                    continue
                
                # Load stamp image
                stamp_img = cv2.imread(os.path.join(stamp['batch'], stamp['file']))  # type: ignore
                if stamp_img is None:
                    continue
                
                # Try different rotations of this stamp
                rotated_stats = get_rotated_stats(stamp_img)
                for stat in rotated_stats:
                    # Calculate distance between stamp and target cell
                    dist = compute_distance(stat['avg_color'], stat['avg_brightness'], cell, cell_brightness)
                    if dist < best_dist:
                        best_dist = dist
                        best_stamp = stamp
                        best_rotation = stat['angle']
                        best_img = stat['img']
            
            # Place the best matching stamp in this cell
            if best_stamp is not None and best_img is not None:
                used.add((best_stamp['batch'], best_stamp['file']))
                
                # Resize stamp to fit cell
                resized_stamp = cv2.resize(best_img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)  # type: ignore
                
                # Calculate position in mosaic
                y1 = r * cell_h
                y2 = y1 + cell_h
                x1 = c * cell_w
                x2 = x1 + cell_w
                
                # Place stamp in mosaic
                mosaic[y1:y2, x1:x2] = resized_stamp
                
                # Record placement in blueprint
                blueprint.append({
                    'row': r+1,
                    'col': c+1,
                    'batch': best_stamp['batch'],
                    'file': best_stamp['file'],
                    'rotation': best_rotation
                })
    
    # Save the completed mosaic
    cv2.imwrite(output_mosaic, mosaic)  # type: ignore
    
    # Export JSON with artwork size info and placement blueprint
    artwork_width_cm = cols * stamp_width_cm
    artwork_height_cm = rows * stamp_height_cm
    export = {
        'artwork_width_cm': artwork_width_cm,
        'artwork_height_cm': artwork_height_cm,
        'stamp_width_cm': stamp_width_cm,
        'stamp_height_cm': stamp_height_cm,
        'blueprint': blueprint
    }
    with open(output_blueprint, 'w') as f:
        json.dump(export, f, indent=2)

if __name__ == '__main__':
    main() 