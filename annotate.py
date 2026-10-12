"""Interactive box annotation into the reviewed clip manifest format."""
import argparse
import json
from pathlib import Path

import cv2


def annotate(video, clip_id, group, split, stride, output, recorded_at='', weather=''):
    if stride < 1:
        raise ValueError('Stride must be positive')
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f'Cannot open {video}')
    annotations = []
    index = 0
    keys = {ord('b'): 'boat', ord('p'): 'person', ord('f'): 'person_fishing'}
    cv2.namedWindow('Annotate', cv2.WINDOW_NORMAL)
    try:
        quit_requested = False
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if index % stride:
                index += 1
                continue
            objects = []
            while True:
                display = image.copy()
                for obj in objects:
                    x, y, w, h = map(int, obj['box'])
                    cv2.rectangle(display, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(display, obj['label'], (x,max(15,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.putText(display, f'Frame {index}: b boat | p person | f fishing | u undo | Enter review | q stop',
                            (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.imshow('Annotate', display)
                key = cv2.waitKey(0) & 255
                if key == ord('q'):
                    quit_requested = True
                    break
                if key == 13:
                    annotations.append({'index': index, 'objects': objects})
                    break
                if key == ord('u') and objects:
                    objects.pop()
                if key in keys:
                    box = cv2.selectROI('Annotate', image, fromCenter=False, showCrosshair=True)
                    if box[2] and box[3]:
                        objects.append({'label': keys[key], 'box': list(map(int, box))})
            if quit_requested:
                break
            index += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()
    if not annotations:
        raise ValueError('No frames reviewed; nothing saved')
    clip = {'id': clip_id, 'video': str(Path(video).resolve()), 'group': group, 'split': split,
            'recorded_at': recorded_at, 'weather': weather, 'frames': annotations}
    with Path(output).open('x') as file:
        json.dump({'clips': [clip]}, file, indent=2)
    return len(annotations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--clip-id', required=True)
    parser.add_argument('--group', required=True, help='Shared source recording/session ID; never span splits')
    parser.add_argument('--split', choices=['train','val','test'], required=True)
    parser.add_argument('--stride', type=int, default=30)
    parser.add_argument('--output', required=True)
    parser.add_argument('--recorded-at', default='')
    parser.add_argument('--weather', default='')
    args = parser.parse_args()
    try:
        print(f'Reviewed frames: {annotate(args.video,args.clip_id,args.group,args.split,args.stride,args.output,args.recorded_at,args.weather)}')
    except (ValueError, OSError) as error:
        parser.exit(2, f'Annotation: {error}\n')


if __name__ == '__main__':
    main()
