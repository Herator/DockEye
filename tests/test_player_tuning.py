import unittest

import numpy as np

from src.mediaPlayer import Player


class TestPausedTuning(unittest.TestCase):
    def test_slider_change_redraws_paused_frame_without_relearning(self):
        class Detector:
            def __init__(self):
                self.setting = 0
                self.updates = 0

            def __call__(self, frame):
                self.updates += 1
                return self.redraw(frame)

            def redraw(self, frame):
                frame[:] = self.setting
                return frame

        detector = Detector()
        player = Player.__new__(Player)
        player.boxes = detector
        player.reprocess = detector.redraw
        player.settings_key = lambda: detector.setting
        player.frame = np.zeros((40, 40, 3), dtype=np.uint8)
        player.i, player.pos = 0, 0
        player.paused = True
        player.warmup_remaining = 3
        player.cache_key, player.cache = None, None
        player.toggles = {'flip': False, 'boxes': True, 'info': False, 'list': False}
        first = player.render()
        self.assertEqual(int(first[0, 0, 0]), 0)
        self.assertEqual(player.warmup_remaining, 2)
        detector.setting = 90
        second = player.render()
        self.assertEqual(int(second[0, 0, 0]), 90)
        self.assertEqual(detector.updates, 1)
        self.assertEqual(player.warmup_remaining, 2)

if __name__ == '__main__':
    unittest.main()
