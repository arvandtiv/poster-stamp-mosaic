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

# Process each batch defined in the metadata
for batch in metadata['batches']:
    # Extract configuration parameters from the batch
    input_image = batch['scan_file']  # Path to the scanned image file
    output_dir = f"extracted_stamps_{os.path.splitext(os.path.basename(input_image))[0]}"  # Create output directory name based on input filename
    rows = batch['rows']  # Number of rows in the stamp grid
    cols = batch['cols']  # Number of columns in the stamp grid
    
    # Get padding values from batch config, with defaults if not specified
    # Padding determines how much to crop from the edges of each stamp
    padding_x = batch.get('padding_x', 23)  # Default to 23px if not specified
    padding_y = batch.get('padding_y', 38)  # Default to 38px if not specified
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the input image using OpenCV
    img = cv2.imread(input_image)  # type: ignore
    if img is None:
        print(f"Image not found: {input_image}")
        continue
    
    # Get image dimensions
    height, width, _ = img.shape
    
    # Calculate the size of each stamp cell in the grid
    stamp_h = height // rows  # Height of each stamp cell
    stamp_w = width // cols   # Width of each stamp cell
    
    # List to store position information for each extracted stamp
    positions = []
    
    # Iterate through each cell in the grid
    for row in range(rows):
        for col in range(cols):
            # Calculate the bounding box for this stamp cell
            y1 = row * stamp_h  # Top edge of the cell
            y2 = (row + 1) * stamp_h  # Bottom edge of the cell
            x1 = col * stamp_w  # Left edge of the cell
            x2 = (col + 1) * stamp_w  # Right edge of the cell
            
            # Apply padding to create a cropped version of the stamp
            # This removes borders and focuses on the stamp content
            y1_c = min(height, max(0, y1 + padding_y))  # Crop from top
            y2_c = min(height, max(0, y2 - padding_y))  # Crop from bottom
            x1_c = min(width, max(0, x1 + padding_x))   # Crop from left
            x2_c = min(width, max(0, x2 - padding_x))   # Crop from right
            
            # Only process if the cropped area is valid (positive dimensions)
            if y2_c > y1_c and x2_c > x1_c:
                # Extract the cropped stamp image
                stamp = img[y1_c:y2_c, x1_c:x2_c]
                
                # Generate filename for this stamp (1-indexed for user-friendly naming)
                filename = f'stamp_r{row+1}_c{col+1}.jpg'
                
                # Save the extracted stamp as a JPEG file
                cv2.imwrite(os.path.join(output_dir, filename), stamp)  # type: ignore
                
                # Record position information for this stamp
                positions.append({'row': row+1, 'col': col+1, 'file': filename})
    
    # Save position metadata to a JSON file
    # This helps track where each stamp was located in the original grid
    with open(os.path.join(output_dir, 'positions.json'), 'w') as f:
        json.dump(positions, f, indent=2) 