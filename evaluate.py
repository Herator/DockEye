"""Evaluate run outputs against manually reviewed test-clip annotations."""
import argparse
import csv
import json
from pathlib import Path

from dataset_tool import load_manifest
from detector import CLASSES, iou


def match_boxes(expected, predicted, threshold=0.5):
    pairs = sorted(((iou(a['box'], b['box']), i, j)
                    for i, a in enumerate(expected) for j, b in enumerate(predicted)
                    if a['label'] == b['label']), reverse=True)
    used_expected, used_predicted = set(), set()
    for overlap, i, j in pairs:
        if overlap >= threshold and i not in used_expected and j not in used_predicted:
            used_expected.add(i)
            used_predicted.add(j)
    return len(used_expected), len(predicted)-len(used_predicted), len(expected)-len(used_expected)


def metrics(tp, fp, fn):
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn,
            'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None}


def evaluate(manifest, clip_id, run_directory):
    data = load_manifest(manifest)
    clip = next((c for c in data['clips'] if c['id'] == clip_id), None)
    if clip is None or clip['split'] != 'test':
        raise ValueError('Select a held-out test clip from the manifest')
    root = Path(run_directory)
    summary = json.loads((root/'summary.json').read_text())
    if summary['mode'] != 'model':
        raise ValueError('Motion-only output cannot evaluate semantic classes')
    predictions = {row['frame']: row for row in
                   (json.loads(line) for line in (root/'predictions.jsonl').read_text().splitlines())}
    with (root/'events.csv').open(newline='') as file:
        events = list(csv.DictReader(file))
    if events and any(Path(event['source']).resolve() != Path(clip['video']).resolve() for event in events):
        raise ValueError('Run source does not match the selected clip')
    if Path(summary['parameters']['video']).resolve() != Path(clip['video']).resolve():
        raise ValueError('Run source does not match the selected clip')
    totals = {label: [0, 0, 0] for label in CLASSES}
    annotations = clip.get('frames', [])
    if not annotations:
        raise ValueError('Clip has no reviewed frame annotations')
    for frame in annotations:
        if frame['index'] not in predictions:
            raise ValueError(f'Annotated frame {frame["index"]} was not processed')
        predicted = [dict(track, label='person' if track['label'] == 'person_fishing_candidate' else track['label'])
                     for track in predictions[frame['index']]['tracks']]
        for label in CLASSES:
            values = match_boxes([o for o in frame['objects'] if o['label'] == label],
                                 [o for o in predicted if o['label'] == label])
            totals[label] = [a+b for a, b in zip(totals[label], values)]
    report = {'clip': clip_id, 'annotated_frames': len(annotations),
              'classes': {label: metrics(*counts) for label, counts in totals.items()},
              'boat_count': summary['boat_count'],
              'manual_boat_count': clip.get('manual_boat_count'),
              'fishing_supported': summary.get('fishing_supported', False),
              'fishing_candidates_supported': summary.get('fishing_candidates_supported', False),
              'note': 'Box metrics are on reviewed frames; retained tracks count as predictions.'}
    if 'events' in clip:
        gt = clip['events']
        used = set()
        tp = 0
        for event in events:
            for index, expected in enumerate(gt):
                if index in used or expected['type'] != event['event_type']:
                    continue
                start, end = float(event['start_seconds']), float(event['end_seconds'])
                if expected['type'] == 'boat_crossing':
                    matches = abs(start-expected['start_seconds']) <= 1.0
                else:
                    overlap = max(0, min(end, expected['end_seconds'])-max(start, expected['start_seconds']))
                    union = max(end, expected['end_seconds'])-min(start, expected['start_seconds'])
                    matches = union > 0 and overlap/union >= 0.5
                if matches:
                    used.add(index)
                    tp += 1
                    break
        report['events'] = metrics(tp, len(events)-tp, len(gt)-tp)
    if clip.get('empty_dock'):
        # Empty status is a manual full-clip ground-truth assertion, not inferred from a sample.
        if any(frame['objects'] for frame in annotations) or clip.get('events'):
            raise ValueError('Empty-dock clip contains positive annotations')
        hours = summary.get('processed_video_seconds',summary['frames_processed']/summary['fps'])/3600
        if hours <= 0:
            raise ValueError('Empty clip duration is zero')
        report['false_events_per_hour'] = len(events)/hours
        report['duration_hours'] = hours
        report['false_track_frames_per_hour'] = sum(len(row['tracks']) for row in predictions.values())/hours
    if clip.get('manual_boat_count') is not None:
        report['boat_count_error'] = summary['boat_count']-clip['manual_boat_count']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        report = evaluate(args.manifest, args.clip, args.run)
        with Path(args.output).open('x') as file:
            json.dump(report, file, indent=2)
        print(json.dumps(report, indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'Evaluation: {error}\n')


if __name__ == '__main__':
    main()
