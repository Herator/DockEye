import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from dataset.classifier import load_dataset, image_features


class DatasetTests(unittest.TestCase):
    def test_duplicate_images_are_excluded_and_class_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'boat').mkdir()
            (root / 'background').mkdir()
            image = np.full((10, 10, 3), 50, dtype=np.uint8)
            cv2.imwrite(str(root / 'boat' / 'one.png'), image)
            cv2.imwrite(str(root / 'boat' / 'two.png'), image)
            cv2.imwrite(str(root / 'background' / 'other.png'), 255 - image)
            samples, labels, duplicates = load_dataset(root)
            self.assertEqual((len(samples), sorted(labels), duplicates), (2, [0, 1], 1))
            cv2.imwrite(str(root / 'background' / 'conflict.png'), image)
            with self.assertRaisesRegex(ValueError, 'conflicting labels'):
                load_dataset(root)

    def test_features_have_consistent_size(self):
        self.assertEqual(image_features(np.zeros((15, 30, 3), np.uint8)).shape,
                         image_features(np.zeros((60, 10, 3), np.uint8)).shape)

if __name__ == '__main__':
    unittest.main()
