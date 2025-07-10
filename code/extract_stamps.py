import cv2
import os
import json

# INSTRUCTIONS:
# Duplicate 'code/batch_metadata_example.json' as 'code/batch_metadata.json'.
# Update 'batch_metadata.json' with details for each scan batch you want to process.

# Load batch metadata
with open('code/batch_metadata.json', 'r') as f:
    metadata = json.load(f)

for batch in metadata['batches']:
    input_image = batch['scan_file']
    output_dir = f"extracted_stamps_{os.path.splitext(os.path.basename(input_image))[0]}"
    rows = batch['rows']
    cols = batch['cols']
    padding_x = batch.get('padding_x', 23)  # Default to 23px if not specified
    padding_y = batch.get('padding_y', 38)  # Default to 38px if not specified
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(input_image)  # type: ignore
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
            y1_c = min(height, max(0, y1 + padding_y))
            y2_c = min(height, max(0, y2 - padding_y))
            x1_c = min(width, max(0, x1 + padding_x))
            x2_c = min(width, max(0, x2 - padding_x))
            if y2_c > y1_c and x2_c > x1_c:
                stamp = img[y1_c:y2_c, x1_c:x2_c]
                filename = f'stamp_r{row+1}_c{col+1}.jpg'
                cv2.imwrite(os.path.join(output_dir, filename), stamp)  # type: ignore
                positions.append({'row': row+1, 'col': col+1, 'file': filename})
    with open(os.path.join(output_dir, 'positions.json'), 'w') as f:
        json.dump(positions, f, indent=2) 