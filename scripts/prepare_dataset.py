"""
Dataset Initializer Script for Plant Disease AI.

Generates data directory structures, stratified split metadata CSVs,
and sample synthetic crop images for local development and unit testing.
"""

import os
import sys
import numpy as np
from PIL import Image

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.dataset_split import DatasetSplitter


def generate_sample_images(output_dir: str = "data/raw/sample_leaves", count: int = 5):
    """Generates synthetic green leaf-like images for local testing."""
    os.makedirs(output_dir, exist_ok=True)
    generated_paths = []

    for i in range(count):
        # Create 256x256 image with green plant leaf distribution
        img_arr = np.zeros((256, 256, 3), dtype=np.uint8)
        # Background: darkish ground/leaf gradient
        img_arr[:, :, 0] = np.random.randint(20, 60, (256, 256))   # Red channel
        img_arr[:, :, 1] = np.random.randint(100, 220, (256, 256)) # Green channel
        img_arr[:, :, 2] = np.random.randint(10, 50, (256, 256))   # Blue channel
        
        # Add simulated leaf spots (brownish/yellowish)
        if i % 2 == 0:
            cx, cy = np.random.randint(50, 200), np.random.randint(50, 200)
            rr, cc = np.ogrid[:256, :256]
            mask = (rr - cx) ** 2 + (cc - cy) ** 2 <= 30 ** 2
            img_arr[mask, 0] = 160 # Red spot
            img_arr[mask, 1] = 120 # Green spot
            img_arr[mask, 2] = 20  # Blue spot

        img = Image.fromarray(img_arr)
        file_path = os.path.join(output_dir, f"sample_plant_{i+1}.jpg")
        img.save(file_path)
        generated_paths.append(file_path)

    # Also create a non-plant sample image (e.g., pure grey noise / concrete texture)
    non_plant_arr = np.random.randint(100, 150, (256, 256, 3), dtype=np.uint8)
    non_plant_path = os.path.join(output_dir, "sample_not_a_plant.jpg")
    Image.fromarray(non_plant_arr).save(non_plant_path)
    generated_paths.append(non_plant_path)

    return generated_paths


def main():
    print("=" * 60)
    print("Initializing Plant Disease Dataset Environment...")
    print("=" * 60)

    # 1. Generate Stratified Split CSVs
    splitter = DatasetSplitter(
        class_mapping_path="data/metadata/plantvillage_class_mapping.csv",
        output_dir="data/processed"
    )
    splits = splitter.create_stratified_split(samples_per_class=100)
    saved_files = splitter.save_splits(splits)

    for split_name, path in saved_files.items():
        print(f"Generated split '{split_name}': {path} ({len(splits[split_name])} records)")

    # 2. Generate Sample Dev Images
    sample_paths = generate_sample_images()
    print(f"Created {len(sample_paths)} test sample images in 'data/raw/sample_leaves/'")
    print("=" * 60)
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
