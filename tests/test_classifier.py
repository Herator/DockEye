import tempfile
import unittest
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from dataset.classifier import load_dataset, image_features


class DatasetTests(unittest.TestCase):
    def test_training_saves_reloadable_model_and_disjoint_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, intensity in [('background', 30), ('boat', 220)]:
                folder = root / name
                folder.mkdir()
                for i in range(6):
                    image = np.full((20, 20, 3), intensity + i, dtype=np.uint8)
                    cv2.imwrite(str(folder / f'{i}.png'), image)
            output = root / 'output'
            subprocess.run([sys.executable, '-m', 'dataset.classifier', '--data', str(root),
                            '--output', str(output)], check=True, capture_output=True, text=True)
            result = json.loads((output / 'evaluation.json').read_text())
            train = set(result['train'])
            test = {item['path'] for item in result['test']}
            self.assertEqual(len(train), 10)
            self.assertEqual(len(test), 2)
            self.assertTrue(train.isdisjoint(test))
            model = cv2.ml.SVM_load(str(output / 'boat_svm.xml'))
            for item in result['test']:
                image = cv2.imread(item['path'])
                _, prediction = model.predict(image_features(image).reshape(1, -1))
                self.assertEqual(int(prediction[0, 0]), item['predicted'])
            self.assertTrue((output / 'review.html').is_file())

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
