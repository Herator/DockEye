"""Offline DockEye operator interface and headless processing CLI."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from detector import ModelDetector, family
from events import EventEngine, EventLogger
from pipeline import Pipeline
from zones import make_zones, draw_the_zone


COLORS = {'person': (0, 0, 255), 'boat': (255, 0, 0)}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--video', default='media/Oscar.mp4')
    result.add_argument('--zones', default='config/MDZones.json')
    result.add_argument('--zone-reference-size', type=int, nargs=2, metavar=('WIDTH','HEIGHT'),
                        help='Explicitly scale existing zones from this size; same camera view only')
    model = result.add_mutually_exclusive_group()
    model.add_argument('--weights', help='Local YOLO weights; fishing requires person_fishing class')
    model.add_argument('--motion-only', action='store_true', help='Diagnostic proposals; no classification/events')
    result.add_argument('--require-fishing', action='store_true')
    result.add_argument('--fishing-method', choices=['model','rod'], default='model',
                        help='rod is an experimental candidate heuristic, not confirmed fishing')
    result.add_argument('--device', default='cpu')
    result.add_argument('--calibration', help='Optional dock-plane homography JSON; not object heights')
    result.add_argument('--confidence', type=float, default=0.5)
    result.add_argument('--headless', action='store_true')
    result.add_argument('--output', help='New run directory; existing directories are never overwritten')
    result.add_argument('--save-video', action='store_true')
    result.add_argument('--fishing-seconds', type=float, default=3)
    result.add_argument('--infer-interval', type=float, default=0.2)
    result.add_argument('--evidence-gap', type=float, default=0.75)
    result.add_argument('--max-gap', type=float, default=1)
    result.add_argument('--warmup', type=float, default=1)
    result.add_argument('--boat-line', type=float, nargs=4, default=[0.5, 0, 0.5, 1],
                        metavar=('X1', 'Y1', 'X2', 'Y2'), help='Counting line in normalized [0,1] coordinates')
    result.add_argument('--boat-direction', choices=['both', 'positive', 'negative'], default='both')
    result.add_argument('--dock-min-area', type=int, default=350)
    result.add_argument('--water-min-area', type=int, default=1000)
    return result


def validate_args(args):
    if not 0 < args.confidence <= 1:
        raise ValueError('Confidence must be in (0, 1]')
    if args.fishing_seconds <= 0 or args.infer_interval <= 0 or args.warmup < 0:
        raise ValueError('Durations must be positive (warmup may be zero)')
    if not args.infer_interval < args.evidence_gap <= args.max_gap:
        raise ValueError('Require infer-interval < evidence-gap <= max-gap')
    if any(not math.isfinite(value) for value in [args.confidence, args.fishing_seconds,
           args.infer_interval, args.evidence_gap, args.max_gap, args.warmup, *args.boat_line]):
        raise ValueError('Parameters must be finite')
    if args.dock_min_area < 1 or args.water_min_area < 1:
        raise ValueError('Minimum areas must be positive')
    if args.zone_reference_size is not None and min(args.zone_reference_size) <= 0:
        raise ValueError('Zone reference dimensions must be positive')
    if any(not 0 <= value <= 1 for value in args.boat_line) or args.boat_line[:2] == args.boat_line[2:]:
        raise ValueError('Boat line must have distinct endpoints within [0, 1]')
    if args.require_fishing and not args.weights:
        raise ValueError('--require-fishing needs --weights')
    if args.fishing_method == 'rod' and (not args.weights or args.require_fishing):
        raise ValueError('Rod candidates need --weights and cannot satisfy --require-fishing')


def create_controls(stage):
    cv2.namedWindow('Parameters', cv2.WINDOW_NORMAL)
    for name, settings in stage.settings.items():
        for field, maximum in [('min_area', 10000), ('threshold', 254), ('variance', 200),
                               ('kernel_size', 31), ('blur', 1)]:
            cv2.createTrackbar(f'{name} {field}', 'Parameters', int(getattr(settings, field)), maximum, lambda value: None)


def read_controls(stage):
    for name, settings in stage.settings.items():
        for field in ['min_area', 'threshold', 'variance', 'kernel_size', 'blur']:
            value = cv2.getTrackbarPos(f'{name} {field}', 'Parameters')
            setattr(settings, field, bool(value) if field == 'blur' else max(1, value))


def draw_events(events):
    image = np.full((300, 850, 3), 25, np.uint8)
    cv2.putText(image, 'Confirmed events (CSV and snapshots in run directory)', (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    for row, event in enumerate(events[-8:], 1):
        text = f'#{event.event_id}  {event.event_type}  ID {event.track_id}  at {event.confirmed_seconds:.1f}s'
        cv2.putText(image, text, (12, 25+row*29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow('Events', image)


def video_timestamp(capture, frame_index, fps, previous=None):
    """Use decoded source PTS, with a monotonic FPS fallback for missing PTS."""
    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
    if (math.isfinite(timestamp) and timestamp >= 0 and
            (timestamp > 0 or frame_index == 0) and
            (previous is None or timestamp > previous)):
        return timestamp
    return max(frame_index/fps, previous+1/fps if previous is not None else 0)


def run(args, detector=None):
    validate_args(args)
    cv2.setNumThreads(2)
    if detector is None and args.weights:
        detector = ModelDetector(args.weights, args.confidence, args.device)
    if args.fishing_method == 'rod':
        from rod_detection import RodAssistedDetector
        detector = RodAssistedDetector(detector)
    if args.require_fishing and not detector.has_fishing:
        raise ValueError('This model has no person_fishing class; train custom weights first')
    if detector is None:
        print('MOTION ONLY: boxes are motion regions, not people/boats; event detection disabled.')
    elif getattr(detector, 'has_fishing_candidates', False):
        print('ROD HEURISTIC: only possible fishing is logged as fishing_candidate; human review required.')
    elif not detector.has_fishing:
        print('GENERAL MODEL: person/boat inference enabled; fishing is unavailable with these weights.')
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise ValueError(f'Cannot open video: {args.video}')
    writer = None
    logger = None
    engine = None
    prediction_file = None
    calibration = None
    try:
        if args.calibration:
            from calibration import Calibration
            calibration = Calibration.load(args.calibration)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video has no valid frame rate; transcode it with a known frame rate')
        skipped = 0
        while True:
            ret, first = capture.read()
            if not ret:
                raise ValueError(f'{args.video} has no non-black frames')
            if first.mean() >= 10:
                break
            skipped += 1
        if not args.headless:
            cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        zones = make_zones(first, COLORS, args.zones, headless=args.headless,
                           reference_size=args.zone_reference_size)
        output = args.output or str(Path('runs') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        logger = EventLogger(output, str(Path(args.video).resolve()))
        height, width = first.shape[:2]
        x1, y1, x2, y2 = args.boat_line
        line = (x1*(width-1), y1*(height-1), x2*(width-1), y2*(height-1))
        engine = EventEngine(logger, args.fishing_seconds, args.evidence_gap, line, args.boat_direction)
        pipeline = Pipeline(zones, detector, engine, args.infer_interval, args.warmup, args.max_gap)
        pipeline.motion.settings['person'].min_area = args.dock_min_area
        pipeline.motion.settings['boat'].min_area = args.water_min_area
        if not args.headless:
            create_controls(pipeline.motion)
        if args.save_video:
            writer = cv2.VideoWriter(str(Path(output)/'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            if not writer.isOpened():
                raise OSError('Could not create annotated video')
        prediction_file = (Path(output)/'predictions.jsonl').open('w')
        frame, frame_index, processed = first, skipped, 0
        paused = False
        started = time.monotonic()
        elapsed_started = started
        previous_timestamp = None
        first_timestamp = None
        while True:
            if not args.headless:
                read_controls(pipeline.motion)
            timestamp = video_timestamp(capture,frame_index,fps,previous_timestamp)
            if first_timestamp is None:
                first_timestamp = timestamp
            previous_timestamp = timestamp
            proposals, tracks, detections = pipeline.process(frame, timestamp)
            prediction_file.write(json.dumps({
                'frame': frame_index, 'seconds': timestamp,
                'detections': [{'box': d.box, 'label': d.label, 'confidence': d.confidence} for d in detections],
                'tracks': [{'id': t.id, 'box': t.box, 'label': t.label, 'observed': t.observed} for t in tracks],
                'ground_positions_m': {t.id: calibration.ground_position(t,frame.shape) for t in tracks} if calibration else {},
                'inferred': pipeline.last_inference == timestamp,
            })+'\n')
            display = frame.copy()
            if detector is None:
                for name, boxes in proposals.items():
                    for x, y, w, h in boxes:
                        cv2.rectangle(display, (x, y), (x+w, y+h), (0,165,255) if name == 'person' else (0,200,0), 2)
            for track in tracks:
                x, y, w, h = track.box
                color = (0,200,0) if family(track.label) == 'boat' else (0,165,255)
                cv2.rectangle(display, (x,y), (x+w,y+h), color, 2)
                cv2.putText(display, f'{track.label} #{track.id}', (x,max(15,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            draw_the_zone(display, zones, COLORS)
            cv2.line(display, (round(line[0]),round(line[1])), (round(line[2]),round(line[3])), (255,255,0), 2)
            caption = f'Video {timestamp:.1f}s | boats {engine.boat_count} | events {len(logger.events)}'
            cv2.putText(display, caption, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            if writer is not None:
                writer.write(display)
            processed += 1
            if not args.headless:
                cv2.imshow('Video', display)
                draw_events(logger.events)
                # Account for processing time; if overloaded, playback slows honestly.
                wait_ms = max(1, round(1000*((timestamp-first_timestamp+1/fps)-(time.monotonic()-started))))
                key = cv2.waitKey(wait_ms) & 255
                if key == ord('q'):
                    break
                if key == ord(' '):
                    paused = True
                    while paused:
                        key = cv2.waitKey(30) & 255
                        read_controls(pipeline.motion)
                        if key == ord('q'):
                            paused = False
                            break
                        if key == ord(' '):
                            paused = False
                    if key == ord('q'):
                        break
                    started = time.monotonic()-(timestamp-first_timestamp+1/fps)
            ret, frame = capture.read()
            if not ret:
                break
            frame_index += 1
        engine.close()
        summary = {
            'mode': 'model' if detector else 'motion_only', 'frames_processed': processed,
            'black_frames_skipped': skipped, 'fps': fps,
            'video_duration_seconds': timestamp+1/fps,
            'processed_video_seconds': timestamp-first_timestamp+1/fps,
            'timestamp_basis': 'source_pts_with_monotonic_fps_fallback',
            'elapsed_wall_seconds': time.monotonic()-elapsed_started,
            'boat_count': engine.boat_count, 'confirmed_events': len(logger.events),
            'fishing_supported': bool(detector and detector.has_fishing),
            'fishing_candidates_supported': bool(detector and getattr(detector, 'has_fishing_candidates', False)),
            'model_calls': getattr(detector, 'calls', 0),
            'weights': args.weights, 'parameters': vars(args),
            'motion_settings': {name: vars(settings) for name, settings in pipeline.motion.settings.items()},
        }
        (Path(output)/'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps({'output': output, **{k: summary[k] for k in ['frames_processed','boat_count','confirmed_events','fishing_supported']}}, indent=2))
        return summary
    finally:
        if engine is not None:
            engine.close()
        if prediction_file is not None:
            prediction_file.close()
        if writer is not None:
            writer.release()
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as error:
        argument_parser.exit(2, f'DockEye: {error}\n')


if __name__ == '__main__':
    main()
