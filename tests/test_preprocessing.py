"""
Unit Tests for Image and Video Preprocessing Pipeline.
"""

import unittest
import os
import tempfile
import numpy as np
from PIL import Image

from src.preprocessing.image_transforms import ImageTransformer
from src.preprocessing.image_validator import ImageValidator
from src.preprocessing.video_extractor import VideoFrameExtractor
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.dataset_split import DatasetSplitter
from src.contracts import PredictionStatus


class TestPreprocessingPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create valid green leaf synthetic image
        green_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        green_arr[:, :, 1] = 200  # Strong green channel
        green_arr[:, :, 0] = 30
        green_arr[:, :, 2] = 20
        self.valid_leaf_path = os.path.join(self.temp_dir.name, "green_leaf.jpg")
        Image.fromarray(green_arr).save(self.valid_leaf_path)

        # Create non-plant grey noise synthetic image
        grey_arr = np.full((100, 100, 3), 128, dtype=np.uint8)
        self.non_plant_path = os.path.join(self.temp_dir.name, "non_plant.jpg")
        Image.fromarray(grey_arr).save(self.non_plant_path)

        # Create tiny image below dimension limit
        tiny_arr = np.zeros((10, 10, 3), dtype=np.uint8)
        self.tiny_path = os.path.join(self.temp_dir.name, "tiny.jpg")
        Image.fromarray(tiny_arr).save(self.tiny_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_image_transformer_resize_and_shape(self):
        transformer = ImageTransformer(target_size=(224, 224), augment=False)
        arr = transformer.transform(self.valid_leaf_path, return_tensor=False)
        self.assertEqual(arr.shape, (3, 224, 224))
        self.assertEqual(arr.dtype, np.float32)

    def test_image_validator_valid_plant(self):
        validator = ImageValidator()
        res = validator.validate(self.valid_leaf_path)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["status"], PredictionStatus.SUPPORTED.value)
        self.assertGreater(res["foliage_ratio"], 0.05)

    def test_image_validator_not_a_plant(self):
        validator = ImageValidator()
        res = validator.validate(self.non_plant_path)
        self.assertFalse(res["is_valid"])
        self.assertEqual(res["status"], PredictionStatus.NOT_A_PLANT.value)

    def test_image_validator_dimension_check(self):
        validator = ImageValidator(min_width=32, min_height=32)
        res = validator.validate(self.tiny_path)
        self.assertFalse(res["is_valid"])
        self.assertIn("dimensions", res["reason"].lower())

    def test_preprocessing_pipeline_end_to_end(self):
        pipeline = PreprocessingPipeline(target_size=(224, 224))
        ok, tensor_data, meta = pipeline.process_image(self.valid_leaf_path, return_tensor=False)
        self.assertTrue(ok)
        assert tensor_data is not None
        self.assertEqual(tensor_data.shape, (3, 224, 224))
        self.assertEqual(meta["status"], PredictionStatus.SUPPORTED.value)

    def test_dataset_splitter(self):
        splitter = DatasetSplitter(
            class_mapping_path="data/metadata/plantvillage_class_mapping.csv",
            output_dir=self.temp_dir.name
        )
        splits = splitter.create_stratified_split(samples_per_class=10)
        self.assertIn("train", splits)
        self.assertIn("val", splits)
        self.assertIn("test", splits)
        self.assertEqual(len(splits["train"]), 38 * 7)  # 70% of 10 = 7 per class


if __name__ == "__main__":
    unittest.main()
