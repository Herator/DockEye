"""Compare a pretrained detector on a few frames from the project videos."""
import argparse
import base64
import html
import json
import os
from pathlib import Path

# Keep the detector's configuration and plots inside the local, ignored model folder.
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path('model/yolo_experiment/config').resolve()))
os.environ.setdefault('MPLCONFIGDIR', str(Path('model/yolo_experiment/matplotlib').resolve()))

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('videos', nargs='+', type=Path)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--weights', type=Path, default=Path('model/yolo11n.pt'))
    parser.add_argument('--output-dir', type=Path, default=Path('model/yolo_samples'))
    args = parser.parse_args()
    if args.samples < 1 or not 0 <= args.confidence <= 1:
        parser.error('samples must be positive and confidence must be between 0 and 1')
    weights = args.weights
    if not weights.is_file():
        parser.error(f'Model file not found: {weights}')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'review.html').exists() or (output / 'results.json').exists():
        parser.error(f'Results already exist in {output}; choose another --output-dir')
    detector = YOLO(str(weights))
    cards, rows = [], []
    for video in args.videos:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open video: {video}')
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        if count < 1:
            raise RuntimeError(f'No frames reported for {video}')
        for i in range(args.samples):
            index = min(count - 1, int((i + 0.5) * count / args.samples))
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'Cannot read frame {index} from {video}')
            # COCO classes 0 and 8 are person and boat. No training happens here.
            result = detector.predict(frame, classes=[0, 8], conf=args.confidence,
                                      imgsz=640, verbose=False, device='cpu')[0]
            detections = [{'class': result.names[int(box.cls.item())],
                           'confidence': round(float(box.conf.item()), 3),
                           'xyxy': [round(float(v), 1) for v in box.xyxy[0].tolist()]}
                          for box in result.boxes]
            plotted = result.plot()
            ok, encoded = cv2.imencode('.jpg', plotted, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError(f'Cannot encode frame {index} from {video}')
            image = base64.b64encode(encoded).decode('ascii')
            desc = ', '.join(f"{d['class']} ({d['confidence']:.0%})" for d in detections) or 'No person or boat detected'
            seconds = index / fps
            rows.append({'video': str(video), 'frame': index, 'seconds': round(seconds, 2), 'detections': detections})
            cards.append(f'<article><h2>{html.escape(video.name)} — {seconds:.1f}s</h2>'
                         f'<p>{html.escape(desc)}</p><img src="data:image/jpeg;base64,{image}" '
                         f'alt="YOLO prediction on {html.escape(video.name, quote=True)} at {seconds:.1f} seconds"></article>')
            print(f'{video.name} {seconds:.1f}s: {desc}', flush=True)
        capture.release()
    (output / 'results.json').write_text(json.dumps({'model': str(weights), 'confidence': args.confidence,
                                                    'frames': rows}, indent=2, ensure_ascii=False) + '\n')
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>YOLO video experiment</title><style>
body{font:16px system-ui,sans-serif;color:#172333;background:#f4f6f8;margin:0}main{max-width:1200px;margin:auto;padding:25px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:18px}article{background:white;border:1px solid #d6dfe6;border-radius:8px;padding:14px}h2{font-size:18px}img{width:100%;height:auto}p{line-height:1.5}.note{background:#fff4db;padding:14px}</style></head><body><main><h1>YOLO video experiment</h1><p>Five sampled frames per video. Coloured boxes are the model's predictions; the percentage is its confidence, not measured accuracy.</p><p class="note">This model was pretrained on general images, not trained on our dock. These sampled frames have not been labelled, so they do not measure accuracy. A missed box may be a missed object. People fishing are not a separate class.</p><div class="grid">CARDS</div></main></body></html>'''
    page = page.replace('Five sampled frames per video.', f'{args.samples} sampled frames per video.')
    (output / 'review.html').write_text(page.replace('CARDS', '\n'.join(cards)), encoding='utf-8')
    print(f'Open {output / "review.html"}')


if __name__ == '__main__':
    main()
