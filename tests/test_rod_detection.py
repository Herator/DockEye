import cv2
import numpy as np

from detector import Detection
from events import EventEngine, EventLogger
from rod_detection import RodAssistedDetector, RodHeuristic
from tracker import ObjectTracker


def scene():
    return np.zeros((240,400,3),np.uint8)


BOX = (150,60,40,120)


def test_slanted_line_from_person_creates_only_candidate():
    image = scene()
    cv2.line(image,(185,120),(285,45),(255,255,255),2)
    assert RodHeuristic().candidate_line(image,BOX) is not None
    class Model:
        calls = 1
        def detect(self, frame, regions):
            return [Detection(BOX,'person',0.9)]
    detector = RodAssistedDetector(Model())
    assert detector.detect(image,[(0,0,400,240)])[0].label == 'person_fishing_candidate'
    assert not detector.has_fishing


def test_horizontal_railing_and_distant_line_not_candidates():
    image = scene()
    cv2.line(image,(20,120),(390,120),(255,255,255),2)
    assert RodHeuristic().candidate_line(image,BOX) is None
    image = scene()
    cv2.line(image,(250,140),(340,70),(255,255,255),2)
    assert RodHeuristic().candidate_line(image,BOX) is None


def test_rod_events_keep_tentative_type_in_csv(tmp_path):
    logger = EventLogger(tmp_path/'run','clip')
    engine = EventEngine(logger,fishing_seconds=0.5)
    tracker = ObjectTracker()
    for timestamp in [0,0.3,0.6]:
        tracks = tracker.update([Detection(BOX,'person_fishing_candidate',0.9)],timestamp)
        engine.update(tracks,timestamp,scene())
    engine.close()
    assert len(logger.events) == 1
    assert logger.events[0].event_type == 'fishing_candidate'
    assert 'fishing_candidate' in (tmp_path/'run/events.csv').read_text()
