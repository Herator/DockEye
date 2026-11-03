"""Train the fishing / not_fishing classifier on sorted person crops.

Validation crops come from whole clips (--val) that are left out of training, because
neighbouring frames are near copies and a random split would inflate the score.

    python -m dataset.train_fishing --val person_fishing_2 person_on_dock_1
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

CLASSES = ('fishing', 'not_fishing')


def split(crops, out, val_clips):
    """Copy crops to out/{train,val}/{class}/ by clip. Returns counts per split and class."""
    shutil.rmtree(out, ignore_errors=True)
    counts = {}
    for cls in CLASSES:
        for path in sorted((crops / cls).glob('*.jpg')):
            part = 'val' if path.name.split('__')[0] in val_clips else 'train'
            (out / part / cls).mkdir(parents=True, exist_ok=True)
            shutil.copy(path, out / part / cls / path.name)
            counts[part, cls] = counts.get((part, cls), 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--val', nargs='+', required=True, help='clip names (file name without extension) used only for validation')
    parser.add_argument('--crops', type=Path, default=Path('dataset/fishing_crops'))
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    data = Path('dataset/fishing_cls')
    counts = split(args.crops, data, set(args.val))
    print('Crops per split:', {f'{p}/{c}': n for (p, c), n in sorted(counts.items())})
    missing = [f'{p}/{c}' for p in ('train', 'val') for c in CLASSES if not counts.get((p, c))]
    if missing:
        raise SystemExit(f'No crops in {missing}. Every class needs crops in both train and val clips.')

    model = YOLO('model/yolo26n-cls.pt')
    model.train(data=str(data), epochs=args.epochs, imgsz=224, project='model/runs', name='fishing', exist_ok=True)
    shutil.copy(model.trainer.best, 'model/fishing_cls.pt')

    # precision / recall for "fishing" on the validation clips
    best = YOLO('model/fishing_cls.pt')
    tp = fp = fn = 0
    for cls in CLASSES:
        for path in sorted((data / 'val' / cls).glob('*.jpg')):
            r = best.predict(str(path), verbose=False)[0]
            said_fishing = r.names[r.probs.top1] == 'fishing'
            tp += said_fishing and cls == 'fishing'
            fp += said_fishing and cls != 'fishing'
            fn += not said_fishing and cls == 'fishing'
    print(f'Fishing on validation clips: precision {tp / max(tp + fp, 1):.2f}, recall {tp / max(tp + fn, 1):.2f} '
          f'(TP {tp}, FP {fp}, FN {fn})')
    print('Saved model/fishing_cls.pt; main.py uses it automatically.')


if __name__ == '__main__':
    main()
