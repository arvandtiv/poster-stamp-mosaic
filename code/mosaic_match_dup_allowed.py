import cv2
import numpy as np
import json
import os

# Parameters
analysis_file = 'all_stamps_analysis.json'
target_image_file = 'target_image_100_100.jpg'
output_mosaic = 'mosaic_output_dup_allowed.jpg'
output_blueprint = 'mosaic_blueprint_dup_allowed.json'
block_size = 3  # Use 3x3 blocks

# Load stamp analysis
def load_stamps():
    with open(analysis_file, 'r') as f:
        return json.load(f)

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
    target = cv2.imread(target_image_file)
    if target is None:
        raise FileNotFoundError(f"Target image not found: {target_image_file}")
    target_h, target_w = target.shape[:2]
    print(f"Target image size: {target_w}x{target_h}")

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
    print(f"Median stamp size: {median_w}x{median_h}")

    # Calculate number of blocks
    n_blocks_y = target_h // block_size
    n_blocks_x = target_w // block_size
    print(f"Blocks: {n_blocks_x}x{n_blocks_y} (block size: {block_size}x{block_size})")

    # Output mosaic size
    mosaic_h = n_blocks_y * median_h
    mosaic_w = n_blocks_x * median_w
    print(f"Output mosaic size: {mosaic_w}x{mosaic_h}")
    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
    blueprint = []

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y1 = by * block_size
            y2 = y1 + block_size
            x1 = bx * block_size
            x2 = x1 + block_size
            block = target[y1:y2, x1:x2]
            avg_color = block.mean(axis=(0, 1))
            avg_brightness = float(cv2.cvtColor(block, cv2.COLOR_BGR2GRAY).mean())
            best_stamp = None
            best_dist = float('inf')
            best_rotation = 0
            best_img = None
            for stamp in stamps:
                stamp_img_path = os.path.join(stamp['batch'], stamp['file'])
                stamp_img = cv2.imread(stamp_img_path)
                if stamp_img is None:
                    print(f"Warning: Could not load stamp image {stamp_img_path}")
                    continue
                rotated_stats = get_rotated_stats(stamp_img)
                for stat in rotated_stats:
                    dist = compute_distance(stat['avg_color'], stat['avg_brightness'], avg_color, avg_brightness)
                    if dist < best_dist:
                        best_dist = dist
                        best_stamp = stamp
                        best_rotation = stat['angle']
                        best_img = stat['img']
            if best_stamp is not None and best_img is not None:
                resized_stamp = cv2.resize(best_img, (median_w, median_h), interpolation=cv2.INTER_AREA)
                my1 = by * median_h
                my2 = my1 + median_h
                mx1 = bx * median_w
                mx2 = mx1 + median_w
                mosaic[my1:my2, mx1:mx2] = resized_stamp
                blueprint.append({
                    'block_y': by,
                    'block_x': bx,
                    'batch': best_stamp['batch'],
                    'file': best_stamp['file'],
                    'rotation': best_rotation
                })
    print("Saving output mosaic and blueprint...")
    cv2.imwrite(output_mosaic, mosaic)
    with open(output_blueprint, 'w') as f:
        json.dump(blueprint, f, indent=2)
    print("Done!")

if __name__ == '__main__':
    main() 