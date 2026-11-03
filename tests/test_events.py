import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.events import EventLog


class EventLogTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        self.csv = root / 'events.csv'
        self.log = EventLog(str(self.csv), str(root / 'snaps'), min_seconds=2)
        self.log.start_video('media/clip.mp4')
        self.frame = np.zeros((100, 100, 3), np.uint8)

    def tearDown(self):
        self.dir.cleanup()

    def run_track(self, cls, tid, start, end):
        for t in np.arange(start, end, 0.5):
            self.log.update(cls, {tid: (10, 10, 20, 20)}, self.frame, t)
        self.log.update(cls, {}, self.frame, end + 2)  # past the 1 s gap

    def rows(self):
        with open(self.csv, newline='') as f:
            return list(csv.DictReader(f))

    def test_confirmed_track_gives_one_row_and_one_snapshot(self):
        self.run_track('boat', 1, 0, 5)
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]['class'], rows[0]['event'], rows[0]['duration_s']), ('boat', 'boat_passed', '4.5'))
        self.assertTrue(Path(rows[0]['snapshot']).is_file())
        self.assertEqual(self.log.count, {'boat': 1})

    def test_short_track_is_ignored(self):
        self.run_track('person', 1, 0, 1.5)
        self.assertFalse(self.csv.exists())
        self.assertEqual(self.log.count, {})

    def test_short_dropout_stays_one_event(self):
        for t in (0, 1, 2, 3):
            self.log.update('boat', {7: (0, 0, 5, 5)}, self.frame, t)
        self.log.update('boat', {}, self.frame, 3.5)  # missing for 0.5 s
        for t in (4, 5):
            self.log.update('boat', {7: (0, 0, 5, 5)}, self.frame, t)
        self.log.update('boat', {}, self.frame, 7)
        rows = self.rows()
        self.assertEqual([(r['id'], r['start_s'], r['end_s']) for r in rows], [('7', '0.0', '5.0')])

    def test_new_video_closes_open_events_and_resets_count(self):
        for t in (0, 1, 2, 3):
            self.log.update('person', {4: (0, 0, 5, 5)}, self.frame, t)
        self.log.start_video('media/next.mp4')
        self.assertEqual([r['video'] for r in self.rows()], ['clip.mp4'])
        self.assertEqual(self.log.count, {})


if __name__ == '__main__':
    unittest.main()
