import cv2
import os
import json

# INSTRUCTIONS:
# Duplicate 'code/batch_metadata_example.json' as 'code/batch_metadata.json'.
# Update 'batch_metadata.json' with details for each scan batch you want to process.

# Load batch metadata from JSON file
# This file contains configuration for each scan batch including grid dimensions and padding
with open('code/batch_metadata.json', 'r') as f:
    metadata = json.load(f)

# Create single output directory for all batches
output_dir = "extracted_stamps"
os.makedirs(output_dir, exist_ok=True)

# List to store position information for all extracted stamps
all_positions = []

# Process each batch defined in the metadata
for batch_idx, batch in enumerate(metadata['batches']):
    # Extract configuration parameters from the batch
    input_image = batch['scan_file']  # Path to the scanned image file
    batch_name = os.path.splitext(os.path.basename(input_image))[0]  # Get batch name from filename
    rows = batch['rows']  # Number of rows in the stamp grid
    cols = batch['cols']  # Number of columns in the stamp grid
    
    # Get padding values from batch config, with defaults if not specified
    # Padding determines how much to crop from the edges of each stamp
    padding_x = batch.get('padding_x', 23)  # Default to 23px if not specified
    padding_y = batch.get('padding_y', 38)  # Default to 38px if not specified
    
    # Get stamp dimensions from batch metadata
    stamp_width_cm = batch.get('stamp_width_cm', 3.0)  # Physical width in cm
    stamp_height_cm = batch.get('stamp_height_cm', 4.5)  # Physical height in cm
    
    # Load the input image using OpenCV
    img = cv2.imread(input_image)  # type: ignore
    if img is None:
        print(f"Image not found: {input_image}")
        continue
    
    # Get image dimensions
    height, width, _ = img.shape
    
    # Use sheet physical size if provided, otherwise fall back to grid*stamp size
    sheet_width_cm = batch.get('sheet_width_cm', cols * stamp_width_cm)
    sheet_height_cm = batch.get('sheet_height_cm', rows * stamp_height_cm)
    
    pixels_per_cm_x = width / sheet_width_cm
    pixels_per_cm_y = height / sheet_height_cm
    
    # Convert stamp dimensions from cm to pixels
    stamp_w_px = int(stamp_width_cm * pixels_per_cm_x)
    stamp_h_px = int(stamp_height_cm * pixels_per_cm_y)
    
    # Calculate the size of each stamp cell in the grid (for reference)
    grid_stamp_h = height // rows  # Height of each stamp cell
    grid_stamp_w = width // cols   # Width of each stamp cell
    
    # --- DEBUG: Draw crop lines on a copy of the input image ---
    debug_img = img.copy()
    crop_color = (0, 255, 0)  # Green in BGR
    crop_thickness = 2
    
    # Iterate through each cell in the grid
    for row in range(rows):
        for col in range(cols):
            # Calculate the bounding box for this stamp cell
            y1 = row * grid_stamp_h  # Top edge of the cell
            y2 = (row + 1) * grid_stamp_h  # Bottom edge of the cell
            x1 = col * grid_stamp_w  # Left edge of the cell
            x2 = (col + 1) * grid_stamp_w  # Right edge of the cell
            
            # Only apply padding to internal stamps
            pad_y1 = 0 if row == 0 else padding_y
            pad_y2 = 0 if row == rows - 1 else padding_y
            pad_x1 = 0 if col == 0 else padding_x
            pad_x2 = 0 if col == cols - 1 else padding_x
            
            y1_c = min(height, max(0, y1 + pad_y1))  # Crop from top
            y2_c = min(height, max(0, y2 - pad_y2))  # Crop from bottom
            x1_c = min(width, max(0, x1 + pad_x1))   # Crop from left
            x2_c = min(width, max(0, x2 - pad_x2))   # Crop from right
            
            # Draw rectangle for this crop area on the debug image
            if y2_c > y1_c and x2_c > x1_c:
                cv2.rectangle(debug_img, (x1_c, y1_c), (x2_c, y2_c), crop_color, crop_thickness)  # type: ignore
                # Extract the cropped stamp image
                stamp = img[y1_c:y2_c, x1_c:x2_c]
                
                # Generate filename for this stamp with batch prefix
                filename = f'{batch_name}_r{row+1}_c{col+1}.jpg'
                
                # Save the extracted stamp as a JPEG file
                cv2.imwrite(os.path.join(output_dir, filename), stamp)  # type: ignore
                
                # Record position information for this stamp
                all_positions.append({
                    'batch': batch_name,
                    'row': row+1,
                    'col': col+1,
                    'file': filename,

                })
    
    # Save the debug image with crop lines
    debug_filename = f"debug_crop_lines_{batch_name}_w{stamp_w_px}_h{stamp_h_px}_wp{padding_x}_hp{padding_y}.jpg"
    cv2.imwrite(os.path.join(output_dir, debug_filename), debug_img)  # type: ignore

# Save aggregated position metadata to a single JSON file
# This helps track where each stamp was located across all batches
with open(os.path.join(output_dir, 'positions.json'), 'w') as f:
    json.dump(all_positions, f, indent=2) 