"""Save crops of every person on the dock, for the fishing classifier.

Afterwards, sort the images in dataset/fishing_crops/unsorted/ by hand into
dataset/fishing_crops/fishing/ and dataset/fishing_crops/not_fishing/ (delete bad ones).
The file name starts with the clip name, so train_fishing.py can split by clip.

    python -m dataset.fishing_crops media/person_fishing_1.mp4 media/person_on_dock_1.mp4
"""
import argparse
import json
from pathlib import Path

import cv2

from src.detector import Detector
from src.fishing import crop


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('videos', nargs='+', type=Path)
    parser.add_argument('--every', type=float, default=1.0, help='seconds between saved frames (neighbouring frames are near copies)')
    parser.add_argument('--zones', type=Path, default=Path('config/MDZones.json'))
    parser.add_argument('--out', type=Path, default=Path('dataset/fishing_crops/unsorted'))
    args = parser.parse_args()

    dock_zone = json.loads(args.zones.read_text())['person']
    args.out.mkdir(parents=True, exist_ok=True)
    for video in args.videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise SystemExit(f'Cannot open {video}')
        step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30) * args.every))
        detector, n, saved = Detector(dock_zone), 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if n % step == 0:
                for tid, box in detector.track(frame)['person'].items():
                    cv2.imwrite(str(args.out / f'{video.stem}__f{n:06d}_id{tid}.jpg'), crop(frame, box))
                    saved += 1
            n += 1
        print(f'{video}: {saved} crops')
    print(f'Now sort {args.out} into fishing/ and not_fishing/ next to it.')


if __name__ == '__main__':
    main()
