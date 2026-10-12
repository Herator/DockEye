"""Event confirmation uses video time, never processing/wall-clock speed."""
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2


@dataclass
class Event:
    event_id: int
    source: str
    track_id: int
    label: str
    event_type: str
    start_seconds: float
    confirmed_seconds: float
    end_seconds: float
    duration_seconds: float
    snapshot: str = ''


class EventLogger:
    def __init__(self, output, source):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / 'snapshots').mkdir()
        self.source = source
        self.events = []
        self.flush()

    def confirm(self, track, event_type, start, timestamp, frame):
        event_id = len(self.events)+1
        filename = f'snapshots/event_{event_id:05d}.jpg'
        if not cv2.imwrite(str(self.output / filename), frame):
            raise OSError('Could not save confirmed-event snapshot')
        event = Event(event_id, self.source, track.id, track.label,
                      event_type, start, timestamp, timestamp, timestamp-start, filename)
        self.events.append(event)
        self.flush()
        return event

    def finish(self, event, timestamp):
        event.end_seconds = max(event.confirmed_seconds, timestamp)
        event.duration_seconds = event.end_seconds-event.start_seconds
        self.flush()

    def flush(self):
        temporary = self.output / 'events.csv.tmp'
        with temporary.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(Event.__dataclass_fields__))
            writer.writeheader()
            writer.writerows(asdict(event) for event in self.events)
        temporary.replace(self.output / 'events.csv')


class EventEngine:
    def __init__(self, logger, fishing_seconds=3.0, evidence_gap=0.75,
                 boat_line=None, boat_direction='both'):
        self.logger = logger
        self.fishing_seconds, self.evidence_gap = fishing_seconds, evidence_gap
        # Line is (x1, y1, x2, y2) in image pixels.
        self.boat_line, self.boat_direction = boat_line, boat_direction
        self.fishing = {}
        self.boat_sides = {}
        self.counted_boats = set()

    @property
    def boat_count(self):
        return len(self.counted_boats)

    def update(self, tracks, timestamp, frame, inference=True):
        for track in tracks:
            if not track.observed:
                continue
            if track.label in ('person_fishing', 'person_fishing_candidate'):
                state = self.fishing.get(track.id)
                if state is not None and timestamp-state['last'] > self.evidence_gap:
                    if state['event'] is not None:
                        self.logger.finish(state['event'], state['last'])
                    state = None
                if state is None:
                    state = {'start': timestamp, 'last': timestamp, 'event': None}
                    self.fishing[track.id] = state
                state['last'] = timestamp
                if state['event'] is None and timestamp-state['start'] >= self.fishing_seconds:
                    state['event'] = self.logger.confirm(
                        track, 'fishing_candidate' if track.label == 'person_fishing_candidate' else 'fishing',
                        state['start'], timestamp, frame)
            if track.label == 'boat' and self.boat_line is not None:
                x1, y1, x2, y2 = self.boat_line
                x, y, w, h = track.box
                cross = (x2-x1)*(y+h/2-y1)-(y2-y1)*(x+w/2-x1)
                # Small dead band avoids counting jitter on the line.
                margin = 3*((x2-x1)**2+(y2-y1)**2)**0.5
                if abs(cross) <= margin:
                    continue
                side = 1 if cross > 0 else -1
                previous = self.boat_sides.get(track.id)
                self.boat_sides[track.id] = side
                direction = 'positive' if side == 1 else 'negative'
                if (previous is not None and previous != side and
                        track.id not in self.counted_boats and
                        self.boat_direction in ('both', direction)):
                    self.counted_boats.add(track.id)
                    self.logger.confirm(track, 'boat_crossing', timestamp, timestamp, frame)
        for tid, state in list(self.fishing.items()):
            track = next((t for t in tracks if t.id == tid), None)
            negative = (inference and track is not None and track.observed and
                        track.label not in ('person_fishing', 'person_fishing_candidate'))
            if negative or timestamp-state['last'] > self.evidence_gap:
                if state['event'] is not None:
                    self.logger.finish(state['event'], state['last'])
                del self.fishing[tid]

    def close(self):
        for state in self.fishing.values():
            if state['event'] is not None:
                self.logger.finish(state['event'], state['last'])
        self.fishing.clear()
