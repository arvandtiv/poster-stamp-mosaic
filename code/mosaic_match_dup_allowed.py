import cv2
import numpy as np
import json
import os

# MOSAIC MATCHING SCRIPT (DUPLICATES ALLOWED)
# This script creates a mosaic by matching individual stamps to target image blocks
# Unlike the main mosaic script, this version allows the same stamp to be used multiple times
# It processes the target image in blocks rather than individual pixels

# Configuration parameters
analysis_file = 'all_stamps_analysis.json'  # File containing stamp analysis data
target_image_file = 'target_image_100_100.jpg'  # Target image to recreate as mosaic
output_mosaic = 'mosaic_output_dup_allowed.jpg'  # Output mosaic image
output_blueprint = 'mosaic_blueprint_dup_allowed.json'  # Blueprint file with placement instructions
block_size = 3  # Size of blocks to process (3x3 pixels per block)

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

def compute_distance(stamp_color, stamp_brightness, target_color, target_brightness):
    """
    Compute similarity distance between a stamp and target block
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
    """Main mosaic creation algorithm with duplicates allowed"""
    # Load all stamp analysis data
    stamps = load_stamps()
    
    # Load target image
    target = cv2.imread(target_image_file)  # type: ignore
    if target is None:
        raise FileNotFoundError(f"Target image not found: {target_image_file}")
    
    # Get target image dimensions
    target_h, target_w = target.shape[:2]
    print(f"Target image size: {target_w}x{target_h}")

    # Determine stamp size for consistency
    # Use the median size of all stamps for consistent cell size
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
    print(f"Median stamp size: {median_w}x{median_h}")

    # Calculate number of blocks in the target image
    # Each block_size x block_size pixels becomes one stamp in the mosaic
    n_blocks_y = target_h // block_size
    n_blocks_x = target_w // block_size
    print(f"Blocks: {n_blocks_x}x{n_blocks_y} (block size: {block_size}x{block_size})")

    # Calculate output mosaic dimensions
    mosaic_h = n_blocks_y * median_h
    mosaic_w = n_blocks_x * median_w
    print(f"Output mosaic size: {mosaic_w}x{mosaic_h}")
    
    # Create blank mosaic canvas
    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
    blueprint = []

    # Process each block in the target image
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            # Extract the current block from target image
            y1 = by * block_size
            y2 = y1 + block_size
            x1 = bx * block_size
            x2 = x1 + block_size
            block = target[y1:y2, x1:x2]
            
            # Calculate average color and brightness for this block
            avg_color = block.mean(axis=(0, 1))
            avg_brightness = float(cv2.cvtColor(block, cv2.COLOR_BGR2GRAY).mean())  # type: ignore
            
            # Find best matching stamp for this block
            best_stamp = None
            best_dist = float('inf')
            best_rotation = 0
            best_img = None
            
            # Try each available stamp (duplicates allowed)
            for stamp in stamps:
                stamp_img_path = os.path.join(stamp['batch'], stamp['file'])
                stamp_img = cv2.imread(stamp_img_path)  # type: ignore
                if stamp_img is None:
                    print(f"Warning: Could not load stamp image {stamp_img_path}")
                    continue
                
                # Try different rotations of this stamp
                rotated_stats = get_rotated_stats(stamp_img)
                for stat in rotated_stats:
                    # Calculate distance between stamp and target block
                    dist = compute_distance(stat['avg_color'], stat['avg_brightness'], avg_color, avg_brightness)
                    if dist < best_dist:
                        best_dist = dist
                        best_stamp = stamp
                        best_rotation = stat['angle']
                        best_img = stat['img']
            
            # Place the best matching stamp in this block position
            if best_stamp is not None and best_img is not None:
                # Resize stamp to fit the mosaic cell
                resized_stamp = cv2.resize(best_img, (median_w, median_h), interpolation=cv2.INTER_AREA)  # type: ignore
                
                # Calculate position in mosaic
                my1 = by * median_h
                my2 = my1 + median_h
                mx1 = bx * median_w
                mx2 = mx1 + median_w
                
                # Place stamp in mosaic
                mosaic[my1:my2, mx1:mx2] = resized_stamp
                
                # Record placement in blueprint
                blueprint.append({
                    'block_y': by,
                    'block_x': bx,
                    'batch': best_stamp['batch'],
                    'file': best_stamp['file'],
                    'rotation': best_rotation
                })
    
    print("Saving output mosaic and blueprint...")
    
    # Save the completed mosaic
    cv2.imwrite(output_mosaic, mosaic)  # type: ignore
    
    # Export JSON with artwork size info and placement blueprint
    artwork_width_cm = n_blocks_x * stamp_width_cm
    artwork_height_cm = n_blocks_y * stamp_height_cm
    export = {
        'artwork_width_cm': artwork_width_cm,
        'artwork_height_cm': artwork_height_cm,
        'stamp_width_cm': stamp_width_cm,
        'stamp_height_cm': stamp_height_cm,
        'blueprint': blueprint
    }
    with open(output_blueprint, 'w') as f:
        json.dump(export, f, indent=2)
    print("Done!")

if __name__ == '__main__':
    main() 