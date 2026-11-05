import time
import unittest
from unittest import mock

import cv2
import numpy as np

import src.mediaPlayer as mp


class FakeCamera:
    """Frame k is a grey image with value k*4, so we can tell frames apart after JPEG."""

    def __init__(self, frames=1000):
        self.k, self.frames = 0, frames

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_COUNT: -1, cv2.CAP_PROP_FPS: 30}.get(prop, 0)

    def read(self):
        if self.k >= self.frames:
            return False, None
        time.sleep(0.002)
        f = np.full((48, 64, 3), self.k * 4 % 256, np.uint8)
        self.k += 1
        return True, f

    def release(self):
        pass


def value(player):
    return round(player.frame.mean() / 4)


class LiveTests(unittest.TestCase):
    def player(self, cam):
        with mock.patch.object(mp, "capture", return_value=cam):
            p = mp.Player(["0"])
        self.addCleanup(p.close)
        return p

    def wait_for(self, p, n):
        for _ in range(500):
            if p.n >= n:
                return
            time.sleep(0.01)
        self.fail(f"camera only gave {p.n} frames")

    def test_rewind_while_recording(self):
        p = self.player(FakeCamera())
        self.assertTrue(p.live)
        self.wait_for(p, 40)
        p.seek(5)
        self.assertEqual((p.pos, value(p)), (5, 5))
        n = p.n
        self.wait_for(p, n + 10)  # still recording while we look at the past
        self.assertTrue(p.next_frame())
        self.assertEqual(value(p), 6)
        p.do("live")
        self.assertTrue(p.at_live_edge())

    def test_old_frames_drop_out_of_the_buffer(self):
        with mock.patch.object(mp, "BUFFER_S", 1):  # 30 frames
            p = self.player(FakeCamera(60))
        p.thread.join()
        p.seek(0)
        self.assertEqual(p.bounds(), (30, 59))
        self.assertEqual((p.pos, value(p)), (30, 30))
        p.seek(59)
        self.assertFalse(p.next_frame())  # camera ended, nothing new

    def test_missing_camera(self):
        p = self.player(FakeCamera(0))
        self.assertEqual(p.frame.shape, (540, mp.W, 3))  # the "Could not open" frame
        self.assertFalse(p.seek(0))


if __name__ == "__main__":
    unittest.main()
