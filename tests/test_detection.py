import unittest

from src.detector import on_dock

DOCK = [(0, 100), (200, 100), (200, 150), (0, 150)]


class DetectionTests(unittest.TestCase):
    def test_on_dock_uses_the_feet(self):
        self.assertTrue(on_dock((50, 60, 20, 60), DOCK))      # feet at y=120, inside
        self.assertFalse(on_dock((50, 160, 20, 30), DOCK))    # below the dock (in a boat)
        self.assertFalse(on_dock((300, 60, 20, 60), DOCK))    # next to the dock


if __name__ == '__main__':
    unittest.main()
