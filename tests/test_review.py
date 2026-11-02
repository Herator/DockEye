import tempfile
import unittest
from pathlib import Path

from dataset.review import write_review


class ReviewTests(unittest.TestCase):
    def test_review_shows_predictions_and_highlights_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'wave.png'
            image.write_bytes(b'example image')
            output = root / 'review.html'
            write_review([{'path': str(image), 'actual': 0, 'predicted': 1}], output)
            page = output.read_text()
            self.assertIn('Actual: Background', page)
            self.assertIn('Predicted: Boat', page)
            self.assertIn('Incorrect', page)
            self.assertIn('data:image/png;base64,', page)

    def test_missing_image_reports_path(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / 'missing.png'
            with self.assertRaisesRegex(FileNotFoundError, 'missing.png'):
                write_review([{'path': str(missing), 'actual': 0, 'predicted': 0}],
                             Path(directory) / 'review.html')
