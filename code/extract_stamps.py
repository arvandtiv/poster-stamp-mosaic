import cv2
import os
import json

# INSTRUCTIONS:
# Duplicate 'code/batch_metadata_example.json' as 'code/batch_metadata.json'.
# Update 'batch_metadata.json' with details for each scan batch you want to process.

# Load batch metadata
with open('code/batch_metadata.json', 'r') as f:
    metadata = json.load(f)

PADDING_X = 15 + 8  # 23px crop from left and right
PADDING_Y = 30 + 8  # 38px crop from top and bottom

for batch in metadata['batches']:
    input_image = batch['scan_file']
    output_dir = f"extracted_stamps_{os.path.splitext(os.path.basename(input_image))[0]}"
    rows = batch['rows']
    cols = batch['cols']
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(input_image)
    if img is None:
        print(f"Image not found: {input_image}")
        continue
    height, width, _ = img.shape
    stamp_h = height // rows
    stamp_w = width // cols
    positions = []
    for row in range(rows):
        for col in range(cols):
            y1 = row * stamp_h
            y2 = (row + 1) * stamp_h
            x1 = col * stamp_w
            x2 = (col + 1) * stamp_w
            # Centered crop with extra crop
            y1_c = min(height, max(0, y1 + PADDING_Y))
            y2_c = min(height, max(0, y2 - PADDING_Y))
            x1_c = min(width, max(0, x1 + PADDING_X))
            x2_c = min(width, max(0, x2 - PADDING_X))
            if y2_c > y1_c and x2_c > x1_c:
                stamp = img[y1_c:y2_c, x1_c:x2_c]
                filename = f'stamp_r{row+1}_c{col+1}.jpg'
                cv2.imwrite(os.path.join(output_dir, filename), stamp)
                positions.append({'row': row+1, 'col': col+1, 'file': filename})
    with open(os.path.join(output_dir, 'positions.json'), 'w') as f:
        json.dump(positions, f, indent=2) 