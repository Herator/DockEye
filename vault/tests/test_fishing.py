import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset.train_fishing import split
from src.fishing import crop


class FishingTests(unittest.TestCase):
    def test_crop_is_padded_and_clipped_to_the_frame(self):
        frame = np.zeros((100, 200, 3), np.uint8)
        self.assertEqual(crop(frame, (90, 40, 20, 20)).shape[:2], (50, 50))   # 15 px pad each side
        self.assertEqual(crop(frame, (0, 0, 20, 20)).shape[:2], (35, 35))     # clipped at the corner

    def test_split_keeps_whole_clips_out_of_training(self):
        with tempfile.TemporaryDirectory() as d:
            crops, out = Path(d) / 'crops', Path(d) / 'out'
            for cls, clip in [('fishing', 'a'), ('fishing', 'b'), ('not_fishing', 'a'), ('not_fishing', 'b')]:
                (crops / cls).mkdir(parents=True, exist_ok=True)
                (crops / cls / f'{clip}__f000001_id1.jpg').write_bytes(b'x')
            counts = split(crops, out, {'b'})
            self.assertEqual(counts, {('train', 'fishing'): 1, ('val', 'fishing'): 1,
                                      ('train', 'not_fishing'): 1, ('val', 'not_fishing'): 1})
            self.assertTrue((out / 'val' / 'fishing' / 'b__f000001_id1.jpg').exists())
            self.assertFalse(any((out / 'train').rglob('b__*')))


if __name__ == '__main__':
    unittest.main()
