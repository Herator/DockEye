"""Experimental boat/background classifier using resized grayscale images and an SVM."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from dataset.review import write_review

LABELS = {'background': 0, 'boat': 1}


def image_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1) / 255.0


def load_dataset(root):
    samples, labels, seen = [], [], {}
    duplicates = 0
    for name, label in LABELS.items():
        folder = root / name
        if not folder.is_dir():
            raise ValueError(f'Missing image folder: {folder}')
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
                continue
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f'Cannot read image: {path}')
            digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            if digest in seen:
                if seen[digest] != label:
                    raise ValueError(f'Image has conflicting labels: {path}')
                duplicates += 1
                continue
            seen[digest] = label
            samples.append((path, image_features(image)))
            labels.append(label)
    return samples, labels, duplicates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('dataset'))
    parser.add_argument('--output', type=Path, default=Path('model'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    samples, labels, duplicates = load_dataset(args.data)
    labels = np.asarray(labels, dtype=np.int32)
    rng = np.random.default_rng(args.seed)
    train, test = [], []
    for label in LABELS.values():
        indices = np.flatnonzero(labels == label)
        if len(indices) < 5:
            raise ValueError('Each class needs at least five unique images')
        indices = rng.permutation(indices)
        count = max(1, round(len(indices) * .2))
        test.extend(indices[:count])
        train.extend(indices[count:])
    features = np.array([feature for _, feature in samples], dtype=np.float32)
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(1.0)
    svm.train(features[train], cv2.ml.ROW_SAMPLE, labels[train])
    _, predicted = svm.predict(features[test])
    predicted = predicted.reshape(-1).astype(int)
    args.output.mkdir(parents=True, exist_ok=True)
    svm.save(str(args.output / 'boat_svm.xml'))
    results = {'seed': args.seed, 'duplicates_removed': duplicates,
               'train': [str(samples[i][0]) for i in train],
               'test': [{'path': str(samples[i][0]), 'actual': int(labels[i]), 'predicted': int(p)}
                        for i, p in zip(test, predicted)],
               'confusion': {f'{actual}_as_{pred}': int(np.sum((labels[test] == actual) & (predicted == pred)))
                             for actual in LABELS.values() for pred in LABELS.values()}}
    (args.output / 'evaluation.json').write_text(json.dumps(results, indent=2) + '\n')
    write_review(results['test'], args.output / 'review.html')
    print('Class labels: 0 = background, 1 = boat')
    print(f'Unique images: {len(samples)}; duplicates removed: {duplicates}; train: {len(train)}; test: {len(test)}')
    print('Confusion counts:', results['confusion'])
    print('Warning: image-level split; similar frames may appear in both sets. This is not an unseen-video evaluation.')
    print(f'Model and split saved in {args.output}')
    print(f'Visual review: {args.output / "review.html"}')


if __name__ == '__main__':
    main()
