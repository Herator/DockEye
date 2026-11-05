import tempfile
import unittest
import runpy
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from src.mediaPlayer import Player


class PlayerResetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = []
        for name, shade in [('one.avi', 30), ('two.avi', 190)]:
            path = Path(self.temp.name) / name
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
            if not writer.isOpened():
                self.fail('Could not create a test video')
            for i in range(15):
                writer.write(np.full((48, 64, 3), shade + i, dtype=np.uint8))
            writer.release()
            self.paths.append(str(path))
        self.state = {}
        def reset():
            self.state['background'] = cv2.createBackgroundSubtractorMOG2()
            self.state['resets'] = self.state.get('resets', 0) + 1
        self.player = Player(self.paths, reset_detection=reset)
        self.addCleanup(lambda: self.player.cap.release())

    def test_seek_restarts_background_and_invalidates_cached_frame(self):
        self.state['background'].apply(self.player.frame)
        self.assertIsNotNone(self.state['background'].getBackgroundImage())
        before = self.state['resets']
        self.player.cache_key = ('old', 'cache')
        self.assertTrue(self.player.seek(5))
        self.assertEqual(self.state['resets'], before + 1)
        self.assertIsNone(self.state['background'].getBackgroundImage())
        self.assertIsNone(self.player.cache_key)
        self.assertGreater(self.player.warmup_remaining, 0)

    def test_switching_video_resets_even_if_frame_number_is_the_same(self):
        self.state['background'].apply(self.player.frame)
        before = self.state['resets']
        self.player.open(1)
        self.assertEqual(self.state['resets'], before + 1)
        self.assertIsNone(self.state['background'].getBackgroundImage())
        self.assertEqual(self.player.i, 1)

    def test_seeking_to_same_frame_does_not_restart_background(self):
        before = self.state['resets']
        self.assertTrue(self.player.seek(self.player.pos))
        self.assertEqual(self.state['resets'], before)

    def test_returning_to_start_resets_after_playback(self):
        for _ in range(5):
            self.assertTrue(self.player.next_frame())
        self.state['background'].apply(self.player.frame)
        before = self.state['resets']
        self.assertTrue(self.player.seek(0))
        self.assertEqual(self.state['resets'], before + 1)
        self.assertIsNone(self.state['background'].getBackgroundImage())

    def test_application_resets_both_real_models_on_seek_and_video_change(self):
        def exercise(player):
            state = player.boxes.__globals__
            player.render()
            first_person, first_boat = state['person_backSub'], state['boat_backSub']
            self.assertIsNotNone(first_person.getBackgroundImage())
            self.assertIsNotNone(first_boat.getBackgroundImage())
            player.seek(5)
            self.assertIsNot(first_person, state['person_backSub'])
            self.assertIsNot(first_boat, state['boat_backSub'])
            self.assertIsNone(state['last_boat_mask'])
            player.render()
            after_seek = state['boat_backSub']
            player.open(1)
            self.assertIsNot(after_seek, state['boat_backSub'])
            self.assertIsNone(state['boat_backSub'].getBackgroundImage())
            player.cap.release()

        with patch('sys.argv', ['main.py', *self.paths]), \
             patch('src.zones.make_zones', return_value={'person': [(0, 0), (63, 0), (63, 47), (0, 47)],
                                                        'boat': [(0, 0), (63, 0), (63, 47), (0, 47)]}), \
             patch.object(cv2, 'namedWindow'), patch.object(cv2, 'destroyWindow'), \
             patch.object(cv2, 'resizeWindow'), patch.object(cv2, 'imshow'), \
             patch.object(cv2, 'createTrackbar'), patch.object(Player, 'run', exercise):
            runpy.run_path('main.py', run_name='__main__')

if __name__ == '__main__':
    unittest.main()
