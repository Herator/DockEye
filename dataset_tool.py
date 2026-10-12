"""Build a YOLO dataset from a manually reviewed, clip-grouped manifest."""
import argparse
import json
import math
from pathlib import Path
import re

import cv2
import yaml

from detector import CLASSES


def load_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    clips = data.get('clips', [])
    if not clips:
        raise ValueError('Manifest needs clips')
    groups, identifiers, sources = {}, set(), {}
    for clip in clips:
        identifier = clip['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', identifier) or identifier in identifiers:
            raise ValueError('Clip IDs must be unique letters/numbers/dashes/underscores')
        identifiers.add(identifier)
        split, group = clip['split'], clip['group']
        if split not in ('train', 'val', 'test') or not isinstance(group, str) or not group:
            raise ValueError('Every clip needs train/val/test split and a source group')
        if group in groups and groups[group] != split:
            raise ValueError(f'Data leakage: source group {group} is in multiple splits')
        groups[group] = split
        video = Path(clip['video'])
        source = str((video if video.is_absolute() else path.parent/video).resolve())
        if source in sources and sources[source] != split:
            raise ValueError('Data leakage: the same video is in multiple splits')
        sources[source] = split
        clip['video'] = source
        indices = set()
        for frame in clip.get('frames', []):
            index = frame['index']
            if not isinstance(index, int) or index < 0 or index in indices:
                raise ValueError('Frame indices must be unique nonnegative integers per clip')
            indices.add(index)
            for obj in frame['objects']:
                if obj['label'] not in CLASSES:
                    raise ValueError(f'Unknown class: {obj["label"]}')
                box = obj['box']
                if len(box) != 4 or not all(math.isfinite(v) for v in box) or min(box[:2]) < 0 or min(box[2:]) <= 0:
                    raise ValueError('Boxes must be finite positive xywh pixel coordinates')
    return data


def build_dataset(manifest_path, output):
    data = load_manifest(manifest_path)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f'Dataset output already exists: {output}')
    # Verify all required splits/classes before writing or decoding.
    for split in ('train', 'val', 'test'):
        present = {obj['label'] for clip in data['clips'] if clip['split'] == split
                   for frame in clip.get('frames', []) for obj in frame['objects']}
        if present != set(CLASSES):
            raise ValueError(f'{split} needs labelled examples of every class; found {sorted(present)}')
    output.mkdir(parents=True)
    provenance = []
    for split in ('train', 'val', 'test'):
        (output/'images'/split).mkdir(parents=True)
        (output/'labels'/split).mkdir(parents=True)
    for clip in data['clips']:
        capture = cv2.VideoCapture(clip['video'])
        try:
            if not capture.isOpened():
                raise ValueError(f'Cannot open {clip["video"]}')
            for annotation in clip.get('frames', []):
                capture.set(cv2.CAP_PROP_POS_FRAMES, annotation['index'])
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f'Cannot read {clip["id"]} frame {annotation["index"]}')
                height, width = image.shape[:2]
                stem = f'{clip["id"]}_{annotation["index"]:08d}'
                lines = []
                for obj in annotation['objects']:
                    x, y, w, h = obj['box']
                    if x+w > width or y+h > height:
                        raise ValueError(f'Box outside image: {stem}')
                    lines.append(f'{CLASSES.index(obj["label"])} {(x+w/2)/width:.8f} {(y+h/2)/height:.8f} {w/width:.8f} {h/height:.8f}')
                if not cv2.imwrite(str(output/'images'/clip['split']/f'{stem}.jpg'), image):
                    raise OSError('Failed to write dataset image')
                (output/'labels'/clip['split']/f'{stem}.txt').write_text('\n'.join(lines))
                provenance.append({'image': stem, 'clip': clip['id'], 'group': clip['group'],
                                   'source': clip['video'], 'split': clip['split']})
        finally:
            capture.release()
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    (output/'dataset.yaml').write_text(yaml.safe_dump({
        'path': str(output), 'train': 'images/train', 'val': 'images/val',
        'test': 'images/test', 'names': list(CLASSES),
    }, sort_keys=False))
    return output/'dataset.yaml'


def validate_dataset(dataset_yaml):
    path = Path(dataset_yaml).resolve()
    config = yaml.safe_load(path.read_text())
    names = config.get('names')
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    if names != list(CLASSES):
        raise ValueError(f'Dataset classes/order must be {CLASSES}')
    root = Path(config.get('path', path.parent))
    if not root.is_absolute():
        root = path.parent/root
    provenance = json.loads((root/'provenance.json').read_text())
    groups, sources = {}, {}
    for item in provenance:
        if item['group'] in groups and groups[item['group']] != item['split']:
            raise ValueError('Dataset provenance leaks source groups across splits')
        groups[item['group']] = item['split']
        if item['source'] in sources and sources[item['source']] != item['split']:
            raise ValueError('Dataset provenance leaks videos across splits')
        sources[item['source']] = item['split']
    for split in ('train', 'val', 'test'):
        images = sorted((root/config[split]).glob('*.jpg'))
        documented = {item['image'] for item in provenance if item['split'] == split}
        if not images or {image.stem for image in images} != documented:
            raise ValueError(f'{split}: image/provenance mismatch')
        present = set()
        for image in images:
            label = root/'labels'/split/f'{image.stem}.txt'
            for line in label.read_text().splitlines():
                values = line.split()
                if len(values) != 5:
                    raise ValueError(f'Invalid YOLO label: {label}')
                cid = int(values[0])
                coords = [float(value) for value in values[1:]]
                if cid not in range(len(CLASSES)) or not all(math.isfinite(v) and 0 <= v <= 1 for v in coords) or min(coords[2:]) <= 0:
                    raise ValueError(f'Invalid YOLO box: {label}')
                present.add(cid)
        if present != set(range(len(CLASSES))):
            raise ValueError(f'{split} lacks a class')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        print(build_dataset(args.manifest, args.output))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'Dataset: {error}\n')


if __name__ == '__main__':
    main()
