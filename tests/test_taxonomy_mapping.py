import csv
import os
import unittest

class TestTaxonomyMapping(unittest.TestCase):
    """
    Validation test suite for Phase 2 Canonical Taxonomy and Class Mappings.
    """

    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.pv_mapping_path = os.path.join(self.project_root, "data", "metadata", "plantvillage_class_mapping.csv")
        self.unified_mapping_path = os.path.join(self.project_root, "data", "metadata", "class_mapping.csv")

    def test_plantvillage_mapping_file_exists(self):
        self.assertTrue(os.path.exists(self.pv_mapping_path), f"File not found: {self.pv_mapping_path}")

    def test_unified_mapping_file_exists(self):
        self.assertTrue(os.path.exists(self.unified_mapping_path), f"File not found: {self.unified_mapping_path}")

    def test_plantvillage_class_count(self):
        with open(self.pv_mapping_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        self.assertEqual(len(rows), 38, f"Expected exactly 38 PlantVillage classes, found {len(rows)}")

    def test_no_duplicate_or_empty_canonical_ids(self):
        with open(self.pv_mapping_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        canonical_ids = [r["canonical_id"].strip() for r in rows]
        
        # Check empty
        empty_ids = [cid for cid in canonical_ids if not cid]
        self.assertEqual(len(empty_ids), 0, f"Found empty canonical IDs: {empty_ids}")

        # Check duplicates
        unique_ids = set(canonical_ids)
        self.assertEqual(len(unique_ids), len(canonical_ids), f"Duplicate canonical IDs found. Total: {len(canonical_ids)}, Unique: {len(unique_ids)}")

    def test_unique_class_ids(self):
        with open(self.pv_mapping_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        class_ids = [int(r["class_id"]) for r in rows]
        self.assertEqual(len(set(class_ids)), len(class_ids), "Duplicate class_id values found.")
        self.assertEqual(sorted(class_ids), list(range(38)), "class_id sequence is not 0..37.")

    def test_required_metadata_fields(self):
        with open(self.pv_mapping_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for idx, row in enumerate(rows):
            self.assertTrue(row["plant"].strip(), f"Missing 'plant' at row {idx}")
            self.assertTrue(row["disease"].strip(), f"Missing 'disease' at row {idx}")
            self.assertIn(row["health_status"].strip(), ["healthy", "diseased"], f"Invalid health_status at row {idx}")
            self.assertTrue(row["original_label"].strip(), f"Missing 'original_label' at row {idx}")
            self.assertEqual(row["source_dataset"].strip(), "PlantVillage", f"Invalid source_dataset at row {idx}")

    def test_total_image_count(self):
        with open(self.pv_mapping_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        total_images = sum(int(r["image_count"]) for r in rows)
        self.assertEqual(total_images, 54305, f"Expected 54,305 total images, got {total_images}")


if __name__ == "__main__":
    unittest.main()
