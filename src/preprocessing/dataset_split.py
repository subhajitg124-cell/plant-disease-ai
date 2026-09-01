"""
Dataset Split Generator for Plant Disease Detection.

Implements stratified train/validation/test split generation across 
PlantVillage and external crop disease dataset metadata.
"""

import os
import csv
import random
from typing import List, Dict, Tuple, Optional


class DatasetSplitter:
    """
    Handles stratified splitting of plant disease dataset records.
    """
    def __init__(
        self,
        class_mapping_path: str = "data/metadata/plantvillage_class_mapping.csv",
        output_dir: str = "data/processed",
        seed: int = 42
    ):
        self.class_mapping_path = class_mapping_path
        self.output_dir = output_dir
        self.seed = seed
        random.seed(self.seed)

    def load_class_mapping(self) -> List[Dict[str, str]]:
        """Loads class metadata records from CSV file."""
        if not os.path.exists(self.class_mapping_path):
            raise FileNotFoundError(f"Class mapping file not found at: {self.class_mapping_path}")

        records = []
        with open(self.class_mapping_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        return records

    def create_stratified_split(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        samples_per_class: int = 100
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Generates stratified dataset samples across train, val, and test splits.
        
        Args:
            train_ratio: Proportion of samples for training (default: 0.70)
            val_ratio: Proportion of samples for validation (default: 0.15)
            test_ratio: Proportion of samples for testing (default: 0.15)
            samples_per_class: Number of synthetic/indexed image records per class.
            
        Returns:
            Dictionary containing 'train', 'val', and 'test' lists of record dicts.
        """
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-4:
            raise ValueError("Split ratios (train + val + test) must sum to 1.0")

        classes = self.load_class_mapping()
        splits: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": []}

        for cls_info in classes:
            class_id = cls_info["class_id"]
            canonical_id = cls_info["canonical_id"]
            plant = cls_info["plant"]
            disease = cls_info["disease"]
            health_status = cls_info.get("health_status", "diseased")

            num_train = int(samples_per_class * train_ratio)
            num_val = int(samples_per_class * val_ratio)
            num_test = samples_per_class - num_train - num_val

            # Generate structured records per class
            for i in range(samples_per_class):
                sample_id = f"{canonical_id}_{i:04d}.jpg"
                if i < num_train:
                    split_key = "train"
                elif i < num_train + num_val:
                    split_key = "val"
                else:
                    split_key = "test"

                splits[split_key].append({
                    "sample_id": sample_id,
                    "class_id": class_id,
                    "canonical_id": canonical_id,
                    "plant": plant,
                    "disease": disease,
                    "health_status": health_status,
                    "relative_path": f"{split_key}/{canonical_id}/{sample_id}",
                    "split": split_key
                })

        return splits

    def save_splits(
        self,
        splits: Dict[str, List[Dict[str, str]]]
    ) -> Dict[str, str]:
        """Saves dataset split records to CSV files in output_dir."""
        os.makedirs(self.output_dir, exist_ok=True)
        saved_files = {}

        fieldnames = [
            "sample_id", "class_id", "canonical_id", "plant", 
            "disease", "health_status", "relative_path", "split"
        ]

        for split_name, records in splits.items():
            file_path = os.path.join(self.output_dir, f"{split_name}.csv")
            with open(file_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
            saved_files[split_name] = file_path

        return saved_files


if __name__ == "__main__":
    splitter = DatasetSplitter()
    splits = splitter.create_stratified_split()
    saved = splitter.save_splits(splits)
    print("Dataset splits successfully generated:")
    for k, v in saved.items():
        print(f" - {k}: {v} ({len(splits[k])} samples)")
