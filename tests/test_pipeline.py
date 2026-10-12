import csv
import json

import cv2
import numpy as np
import pytest

from detector import Detection, iou, in_zone, suppress_duplicates
from events import EventEngine, EventLogger
from main import parser, run, video_timestamp
from motionDetection import motion_detection
from pipeline import Pipeline
from tracker import ObjectTracker
from zones import validate_zones


ZONES = {'person': [[0,0],[159,0],[159,119],[0,119]],
         'boat': [[160,0],[319,0],[319,119],[160,119]]}


def image():
    return np.full((120,320,3), 50, np.uint8)


class Shadows:
    def apply(self, frame):
        result = np.zeros(frame.shape[:2], np.uint8)
        result[10:30,10:30] = 127
        result[50:75,50:75] = 255
        return result

    def getShadowValue(self):
        return 127


def test_shadow_removal_at_low_water_threshold():
    boxes = motion_detection(image(), ZONES['person'], 10, 50, Shadows(), np.ones((1,1),np.uint8))
    assert boxes == [(50,50,25,25)]


def test_class_change_preserves_id_and_matching_is_one_to_one():
    tracker = ObjectTracker(max_distance=5)
    first = tracker.update([Detection((10,10,20,40),'person',0.9), Detection((70,10,20,40),'person',0.8)], 0)
    ids = [t.id for t in first]
    changed = tracker.update([Detection((12,10,20,40),'person_fishing',0.9), Detection((72,10,20,40),'person',0.8)], 0.2)
    assert [t.id for t in changed] == ids
    assert changed[0].label == 'person_fishing'
    assert len(tracker.update([], 0.5)) == 2
    assert tracker.update([], 2) == []


def test_fishing_confirmation_single_snapshot_and_final_duration(tmp_path):
    logger = EventLogger(tmp_path/'run', 'clip.mp4')
    engine = EventEngine(logger, fishing_seconds=1, evidence_gap=0.6)
    tracker = ObjectTracker()
    for timestamp in [0,0.5,1,1.5,2]:
        tracks = tracker.update([Detection((10,10,20,40),'person_fishing',0.9)], timestamp)
        engine.update(tracks, timestamp, image())
    engine.close()
    assert len(logger.events) == 1
    assert logger.events[0].duration_seconds == 2
    assert len(list((tmp_path/'run/snapshots').glob('*.jpg'))) == 1
    with (tmp_path/'run/events.csv').open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1 and float(rows[0]['duration_seconds']) == 2


@pytest.mark.parametrize('negative_label', ['person', 'boat'])
def test_negative_classes_never_confirm_fishing(tmp_path, negative_label):
    logger = EventLogger(tmp_path/'run', 'clip')
    engine = EventEngine(logger, fishing_seconds=0.5)
    tracker = ObjectTracker()
    for timestamp in range(4):
        engine.update(tracker.update([Detection((10,10,20,40),negative_label,0.9)], timestamp), timestamp, image())
    engine.close()
    assert not logger.events
    assert not list((tmp_path/'run/snapshots').iterdir())


def test_long_gap_and_negative_observation_reset_confirmation(tmp_path):
    logger = EventLogger(tmp_path/'run', 'clip')
    engine = EventEngine(logger, fishing_seconds=1, evidence_gap=0.6)
    tracker = ObjectTracker(max_gap=3)
    for timestamp, label in [(0,'person_fishing'),(0.5,'person_fishing'),
                             (1.3,'person_fishing'),(1.5,'person'),(1.7,'person_fishing'),(2,'person_fishing')]:
        engine.update(tracker.update([Detection((10,10,20,40),label,0.9)],timestamp),timestamp,image())
    assert not logger.events


def test_boat_crossing_counted_once_per_id(tmp_path):
    logger = EventLogger(tmp_path/'run', 'clip')
    engine = EventEngine(logger, boat_line=(220,0,220,120))
    tracker = ObjectTracker()
    for timestamp, x in [(0,180),(0.2,230),(0.4,180),(0.6,230)]:
        engine.update(tracker.update([Detection((x,30,20,20),'boat',0.9)],timestamp),timestamp,image())
    assert engine.boat_count == 1
    assert len(logger.events) == 1
    assert logger.events[0].event_type == 'boat_crossing'


def test_boat_wrong_direction_and_no_crossing_not_counted(tmp_path):
    logger = EventLogger(tmp_path/'run', 'clip')
    engine = EventEngine(logger, boat_line=(220,0,220,120), boat_direction='positive')
    tracker = ObjectTracker()
    for timestamp, x in [(0,180),(0.2,185),(0.4,230)]:
        engine.update(tracker.update([Detection((x,30,20,20),'boat',0.9)],timestamp),timestamp,image())
    assert engine.boat_count == 0


class FakeDetector:
    has_fishing = True
    calls = 0

    def detect(self, frame, regions):
        self.calls += 1
        return [Detection((30,20,20,40),'person_fishing',0.99)]


def test_active_track_is_classified_after_motion_stops(tmp_path):
    detector = FakeDetector()
    logger = EventLogger(tmp_path/'run', 'clip')
    engine = EventEngine(logger, fishing_seconds=0.6)
    pipeline = Pipeline(ZONES,detector,engine,infer_interval=0.1,warmup_seconds=0)
    class Motion:
        zones = ZONES
        calls = 0
        def process(self, frame):
            self.calls += 1
            return {'person': [(30,20,20,40)] if self.calls == 1 else [], 'boat': []}
    pipeline.motion = Motion()
    for timestamp in [0,0.2,0.4,0.6,0.8]:
        pipeline.process(image(),timestamp)
    assert detector.calls == 5
    assert len(logger.events) == 1


def test_empty_scene_does_not_call_model():
    detector = FakeDetector()
    detector.calls = 0
    pipeline = Pipeline(ZONES,detector,warmup_seconds=0)
    class Motion:
        zones = ZONES
        def process(self, frame):
            return {'person': [], 'boat': []}
    pipeline.motion = Motion()
    pipeline.process(image(),0)
    assert detector.calls == 0


def test_zone_filters_and_duplicate_suppression():
    assert in_zone(Detection((20,20,10,30),'person_fishing',0.9),ZONES)
    assert not in_zone(Detection((200,20,10,30),'person',0.9),ZONES)
    boxes = [Detection((20,20,10,30),'person',0.7),Detection((20,20,10,30),'person_fishing',0.9)]
    assert suppress_duplicates(boxes) == [boxes[1]]
    assert iou(boxes[0].box,boxes[1].box) == 1
    with pytest.raises(ValueError, match='do not fit'):
        validate_zones(ZONES,np.zeros((50,50,3),np.uint8))


def write_video(path, count=40):
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,120))
    assert writer.isOpened()
    for index in range(count):
        frame = np.zeros((120,320,3),np.uint8) if index < 2 else image()
        if index >= 15:
            cv2.rectangle(frame,(30,20),(50,60),(0,0,240),-1)
        writer.write(frame)
    writer.release()


def test_video_to_model_tracking_event_csv_snapshot_and_export(tmp_path):
    video = tmp_path/'clip.mp4'
    write_video(video)
    zones = tmp_path/'zones.json'
    zones.write_text(json.dumps(ZONES))
    args = parser().parse_args(['--video',str(video),'--zones',str(zones),
                               '--headless','--output',str(tmp_path/'run'),
                               '--warmup','0','--fishing-seconds','0.6','--save-video'])
    summary = run(args,detector=FakeDetector())
    assert summary['frames_processed'] == 38
    assert summary['black_frames_skipped'] == 2
    assert summary['confirmed_events'] == 1
    assert summary['fishing_supported']
    capture = cv2.VideoCapture(str(tmp_path/'run/annotated.mp4'))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 38
    capture.release()
    assert len((tmp_path/'run/predictions.jsonl').read_text().splitlines()) == 38
    assert len(list((tmp_path/'run/snapshots').glob('*.jpg'))) == 1
    with pytest.raises(FileExistsError):
        run(args,detector=FakeDetector())


def test_missing_video_is_an_actionable_error(tmp_path):
    args = parser().parse_args(['--video',str(tmp_path/'absent.mp4'),'--headless'])
    with pytest.raises(ValueError, match='Cannot open video'):
        run(args)


def test_variable_frame_timestamps_and_missing_pts_fallback():
    class Capture:
        def __init__(self,milliseconds): self.milliseconds=milliseconds
        def get(self,prop): return self.milliseconds
    assert video_timestamp(Capture(700),3,10,0.2) == 0.7
    assert video_timestamp(Capture(0),2,10) == 0.2
    assert video_timestamp(Capture(500),4,10,0.7) == pytest.approx(0.8)
