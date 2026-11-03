import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('ultralytics'), 'optional YOLO dependency not installed')
class YoloVideoTests(unittest.TestCase):
    def test_reported_frame_count_mismatch_keeps_decoded_frames_and_warns(self):
        from experiments import yolo_video

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, weights = root / 'source.avi', root / 'output.mp4', root / 'weights.pt'
            weights.write_bytes(b'placeholder for mocked detector')
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            for _ in range(4):
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()
            real_capture = cv2.VideoCapture

            class ReportExtraFrame:
                def __init__(self, path):
                    self.capture = real_capture(path)
                def isOpened(self):
                    return self.capture.isOpened()
                def get(self, prop):
                    value = self.capture.get(prop)
                    return value + 1 if prop == cv2.CAP_PROP_FRAME_COUNT else value
                def read(self):
                    return self.capture.read()
                def release(self):
                    self.capture.release()

            class EmptyDetector:
                def predict(self, frame, **kwargs):
                    return [SimpleNamespace(boxes=[], plot=lambda: frame)]

            args = ['yolo_video.py', str(source), '--output', str(output), '--weights', str(weights)]
            printed = io.StringIO()
            with patch('sys.argv', args), patch.object(yolo_video.cv2, 'VideoCapture', ReportExtraFrame), \
                 patch.object(yolo_video, 'YOLO', return_value=EmptyDetector()), contextlib.redirect_stdout(printed):
                yolo_video.main()
            result = json.loads(output.with_suffix('.json').read_text())
            self.assertEqual(result['frames_processed'], 4)
            self.assertEqual(result['frames_reported'], 5)
            self.assertIn('Warning', printed.getvalue())
            self.assertTrue(output.exists())

if __name__ == '__main__':
    unittest.main()
