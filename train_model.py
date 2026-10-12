"""Fine-tune a detector and evaluate once on held-out test clips."""
import argparse
import json
from pathlib import Path

from dataset_tool import validate_dataset
from detector import CLASSES


def train(data, base_model, output, epochs=50, image_size=640, batch=4, device='cpu'):
    validate_dataset(data)
    if not Path(base_model).is_file():
        raise ValueError('Supply local pretrained YOLO weights with --base-model')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Training output exists; choose a new directory')
    from ultralytics import YOLO
    model = YOLO(str(base_model))
    model.train(data=str(Path(data).resolve()), epochs=epochs, imgsz=image_size,
                batch=batch, device=device, workers=0, seed=42, deterministic=True,
                project=str(output.parent), name=output.name, exist_ok=False,
                plots=False, amp=False)
    best = output/'weights/best.pt'
    if not best.is_file():
        raise RuntimeError('Training did not produce best.pt')
    model = YOLO(str(best))
    metrics = model.val(data=str(Path(data).resolve()), split='test', device=device,
                        imgsz=image_size, batch=batch, workers=0, plots=False,
                        project=str(output), name='heldout_test')
    per_class = {}
    for index, cid in enumerate(metrics.box.ap_class_index):
        per_class[CLASSES[int(cid)]] = {
            'precision': float(metrics.box.p[index]), 'recall': float(metrics.box.r[index]),
            'map50': float(metrics.box.ap50[index]), 'map50_95': float(metrics.box.ap[index]),
        }
    report = {'weights': str(best), 'split': 'test', 'classes': per_class,
              'note': 'Ultralytics detection metrics; these do not measure fishing-event accuracy.'}
    (output/'test_metrics.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if min(args.epochs, args.image_size, args.batch) <= 0:
        parser.error('epochs, image-size and batch must be positive')
    try:
        print(json.dumps(train(args.data, args.base_model, args.output, args.epochs,
                               args.image_size, args.batch, args.device), indent=2))
    except (ValueError, OSError) as error:
        parser.exit(2, f'Training: {error}\n')


if __name__ == '__main__':
    main()
