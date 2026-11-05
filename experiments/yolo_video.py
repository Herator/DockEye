"""Run the pretrained model on every frame of a video and draw its boxes."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('YOLO_CONFIG_DIR', str(Path('model/yolo_experiment/config').resolve()))
os.environ.setdefault('MPLCONFIGDIR', str(Path('model/yolo_experiment/matplotlib').resolve()))

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--weights', type=Path, default=Path('model/yolo11n.pt'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    output_json = args.output.with_suffix('.json')
    if output_json.exists():
        parser.error(f'Predictions file already exists: {output_json}')
    if not 0 <= args.confidence <= 1:
        parser.error('Confidence must be between 0 and 1')
    weights = args.weights
    if not weights.is_file():
        parser.error(f'Model file not found: {weights}')
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        parser.error(f'Cannot open video: {args.video}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError('Video is missing a valid frame rate or dimensions')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f'Cannot create output video: {args.output}')
    detector = YOLO(str(weights))
    frames = []
    start = time.monotonic()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # COCO class 0 is person, 8 is boat. This does not train the model.
            result = detector.predict(frame, classes=[0, 8], conf=args.confidence,
                                      imgsz=640, verbose=False, device='cpu')[0]
            detections = [{'class': result.names[int(box.cls.item())],
                           'confidence': round(float(box.conf.item()), 3),
                           'xyxy': [round(float(v), 1) for v in box.xyxy[0].tolist()]}
                          for box in result.boxes]
            writer.write(result.plot())
            frames.append({'frame': len(frames), 'seconds': round(len(frames) / fps, 3),
                           'detections': detections})
            if len(frames) % 100 == 0:
                print(f'Processed {len(frames)} / {total or "?"} frames', flush=True)
    finally:
        capture.release()
        writer.release()
    if not frames:
        raise RuntimeError('No video frames could be read')
    output_json.write_text(json.dumps({'source': str(args.video), 'model': str(weights),
                                      'confidence_threshold': args.confidence, 'image_size': 640,
                                      'fps': fps, 'frames_processed': len(frames), 'frames_reported': total,
                                      'processing_seconds': round(time.monotonic() - start, 2),
                                      'frames': frames}, indent=2) + '\n')
    if total and len(frames) != total:
        print(f'Warning: decoded {len(frames)} of {total} reported frames. '
              'The source may have inaccurate metadata or an incomplete decode; check the video.', flush=True)
    print(f'Processed {len(frames)} frames. Video: {args.output}; predictions: {output_json}', flush=True)


if __name__ == '__main__':
    main()
