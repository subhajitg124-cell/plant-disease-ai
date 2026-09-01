"""
Unit Tests for Visual Embedding Extraction Module.
"""

import os
import sys
import unittest

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image
except ImportError:
    Image = None

from src.embeddings.visual_embeddings import VisualEmbeddingExtractor


class TestVisualEmbeddings(unittest.TestCase):

    def setUp(self):
        self.extractor = VisualEmbeddingExtractor(embedding_dim=128)

        # Create green synthetic leaf array
        if np is not None:
            self.leaf_arr = np.zeros((224, 224, 3), dtype=np.uint8)
            self.leaf_arr[:, :, 1] = 200
            self.leaf_arr[:, :, 0] = 30
        else:
            self.leaf_arr = [[0, 200, 0]] * (224 * 224)

    def test_embedding_extraction_dimension(self):
        vec = self.extractor.extract(self.leaf_arr)
        self.assertEqual(len(vec), 128)
        self.assertIsInstance(vec, list)
        self.assertIsInstance(vec[0], float)

    def test_embedding_l2_norm(self):
        vec = self.extractor.extract(self.leaf_arr)
        if np is not None:
            norm = np.linalg.norm(vec)
        else:
            norm = (sum(x**2 for x in vec)) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_batch_extraction(self):
        batch = [self.leaf_arr, self.leaf_arr]
        vecs = self.extractor.extract_batch(batch)
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), 128)


if __name__ == "__main__":
    unittest.main()
