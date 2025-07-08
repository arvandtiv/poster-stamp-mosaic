import cv2
import numpy as np
import json
import os

# Parameters
analysis_file = 'all_stamps_analysis.json'
target_image_file = 'target_image_100_100.jpg'
output_mosaic = 'mosaic_output.jpg'
output_blueprint = 'mosaic_blueprint.json'

# Load batch metadata for physical stamp size
with open('code/batch_metadata.json', 'r') as f:
    batch_metadata = json.load(f)
stamp_width_cm = batch_metadata['batches'][0]['stamp_width_cm']
stamp_height_cm = batch_metadata['batches'][0]['stamp_height_cm']

# Load stamp analysis
def load_stamps():
    with open(analysis_file, 'r') as f:
        return json.load(f)

# Load and resize target image
def load_and_resize_target(rows, cols):
    img = cv2.imread(target_image_file)
    if img is None:
        raise FileNotFoundError(f"Target image not found: {target_image_file}")
    return cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA)

# Compute distance between stamp and target cell (combo of color and brightness)
def compute_distance(stamp_color, stamp_brightness, target_color, target_brightness):
    color_dist = np.linalg.norm(np.array(stamp_color) - np.array(target_color))
    brightness_dist = abs(stamp_brightness - target_brightness)
    return color_dist + brightness_dist  # Simple sum, can be weighted

# Get rotated image and its color/brightness
def get_rotated_stats(img):
    stats = []
    for k, angle in enumerate([0, 90, 180, 270]):
        rotated = np.ascontiguousarray(np.rot90(img, k=k))
        avg_color = rotated.mean(axis=(0, 1))
        avg_brightness = float(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY).mean())
        stats.append({'img': rotated, 'angle': angle, 'avg_color': avg_color, 'avg_brightness': avg_brightness})
    return stats

# Main matching logic
def main():
    stamps = load_stamps()
    rows = max(s['row'] for s in stamps)
    cols = max(s['col'] for s in stamps)
    target = load_and_resize_target(rows, cols)
    blueprint = []
    used = set()

    # Determine cell size for the mosaic
    # Use the median size of all stamps for consistency
    stamp_shapes = []
    for stamp in stamps:
        stamp_img = cv2.imread(os.path.join(stamp['batch'], stamp['file']))
        if stamp_img is not None:
            stamp_shapes.append(stamp_img.shape[:2])
    if not stamp_shapes:
        raise ValueError("No valid stamp images found.")
    median_h = int(np.median([s[0] for s in stamp_shapes]))
    median_w = int(np.median([s[1] for s in stamp_shapes]))

    cell_h, cell_w = median_h, median_w
    mosaic = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            cell = target[r, c]
            cell_brightness = float(np.mean(cell))
            best_stamp = None
            best_dist = float('inf')
            best_rotation = 0
            best_img = None
            for stamp in stamps:
                if (stamp['batch'], stamp['file']) in used:
                    continue  # Use each stamp only once
                stamp_img = cv2.imread(os.path.join(stamp['batch'], stamp['file']))
                if stamp_img is None:
                    continue
                rotated_stats = get_rotated_stats(stamp_img)
                for stat in rotated_stats:
                    dist = compute_distance(stat['avg_color'], stat['avg_brightness'], cell, cell_brightness)
                    if dist < best_dist:
                        best_dist = dist
                        best_stamp = stamp
                        best_rotation = stat['angle']
                        best_img = stat['img']
            if best_stamp is not None and best_img is not None:
                used.add((best_stamp['batch'], best_stamp['file']))
                resized_stamp = cv2.resize(best_img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                y1 = r * cell_h
                y2 = y1 + cell_h
                x1 = c * cell_w
                x2 = x1 + cell_w
                mosaic[y1:y2, x1:x2] = resized_stamp
                blueprint.append({
                    'row': r+1,
                    'col': c+1,
                    'batch': best_stamp['batch'],
                    'file': best_stamp['file'],
                    'rotation': best_rotation
                })
    cv2.imwrite(output_mosaic, mosaic)
    # Export JSON with artwork size info
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