import unittest

import cv2
import numpy as np

from src.motionDetection import boxes_from_foreground


class MotionTuningTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((50, 100), np.uint8)
        self.mask[5:25, 5:25] = 127
        self.mask[5:25, 60:80] = 255
        self.zone = [(0, 0), (99, 0), (99, 49), (0, 49)]
        self.kernel = np.ones((1, 1), np.uint8)

    def test_shadow_only_finds_grey_region_and_threshold_finds_both(self):
        shadow_boxes, shadow_mask = boxes_from_foreground(self.mask, self.zone, 50, 50, self.kernel, True)
        self.assertEqual(shadow_boxes, [(5, 5, 20, 20)])
        self.assertEqual(int(shadow_mask[10, 70]), 0)
        all_boxes, _ = boxes_from_foreground(self.mask, self.zone, 50, 50, self.kernel, False)
        self.assertCountEqual(all_boxes, [(5, 5, 20, 20), (60, 5, 20, 20)])
        foreground_boxes, _ = boxes_from_foreground(self.mask, self.zone, 50, 150, self.kernel, False)
        self.assertEqual(foreground_boxes, [(60, 5, 20, 20)])

    def test_minimum_area_filters_small_regions(self):
        boxes, _ = boxes_from_foreground(self.mask, self.zone, 400, 50, self.kernel, False)
        self.assertEqual(boxes, [])


if __name__ == '__main__':
    unittest.main()
