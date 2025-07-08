# Visual Stamp Mosaic Project

## Overview
This project creates large-scale visual mosaics using scanned images of postage stamps. Each stamp is analyzed for its color and brightness profile, and then used as a "tile" to recreate a target image. The system supports batch scanning of stamps, color/brightness analysis, rotation matching, and blueprint generation for physical assembly.

## Workflow
1. **Batch Metadata Preparation**
   - Describe each batch of scanned stamps in a JSON file (see `code/batch_metadata_example.json`).
   - Specify the scan file name, grid size (rows, cols), and real-world stamp size.

2. **Stamp Extraction**
   - Run `code/extract_stamps.py` to extract individual stamps from each batch based on the metadata.
   - Output: Folders like `extracted_stamps_<batchname>` with individual stamp images and a `positions.json` file.

3. **Stamp Analysis**
   - Run `code/analyze_stamps.py` to analyze all extracted stamps for average color and brightness.
   - Output: `all_stamps_analysis.json` containing analysis for all available stamps.

4. **Mosaic Generation**
   - Run `code/mosaic_match.py` for a grid-based mosaic (no duplication, one stamp per cell).
   - Run `code/mosaic_match_dup_allowed.py` for a pixel-perfect mosaic (stamp duplication allowed, output size is a multiple of the target image size).
   - Both scripts support rotation matching (0°, 90°, 180°, 270°) for best visual fit.
   - Output: Mosaic image (`mosaic_output.jpg` or `mosaic_output_dup_allowed.jpg`) and a blueprint JSON for physical assembly.

## How It Works
- **Extraction:** Each scanned batch is divided into a grid, and each stamp is saved as a separate image with its position recorded.
- **Analysis:** Each stamp's average color and brightness are computed for matching.
- **Mosaic Matching:**
  - For each cell (or pixel) in the target image, the best-matching stamp (and rotation) is selected based on color/brightness similarity.
  - The output mosaic is constructed by placing the chosen (rotated and resized) stamp at the corresponding position.
  - The blueprint JSON records which stamp (and rotation) to use for each position.

## Customization
- Add more batches by updating the batch metadata JSON and rerunning the extraction/analysis scripts.
- Change the target image by replacing `target_image_100_100.jpg`.
- Adjust the mosaic style by choosing which script to run (with or without duplication).

## Requirements
- Python 3
- OpenCV (`opencv-python`)
- NumPy

## Example Usage
```bash
# Activate virtual environment
source venv/bin/activate

# Extract stamps from batches
ython3 code/extract_stamps.py

# Analyze all extracted stamps
python3 code/analyze_stamps.py

# Generate a grid-based mosaic (no duplication)
python3 code/mosaic_match.py

# Generate a pixel-perfect mosaic (duplication allowed)
python3 code/mosaic_match_dup_allowed.py
```

## Output
- `mosaic_output.jpg` or `mosaic_output_dup_allowed.jpg`: The final mosaic image.
- `mosaic_blueprint.json` or `mosaic_blueprint_dup_allowed.json`: Blueprint for physical assembly, including rotation info.

## Notes
- The output mosaic can be very large if the target image is large and/or the stamps are high resolution.
- The system is designed to be extensible for more batches and different target images. 