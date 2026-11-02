# DockEye

DockEye runs motion detection on recorded footage using separate dock and water zones. The coloured boxes indicate the zone where motion was found; they are not object classifications.

## Setup and playback

Use Python 3.10 or newer. From the project directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Place video files in `media/`, or pass paths to `main.py`. Video files are not included in Git. Zone configuration is saved in `config/MDZones.json`.

## Experimental boat classifier

The independent baseline uses OpenCV's linear SVM on 32×32 grayscale images in `dataset/boat/` and `dataset/background/`. It removes exact decoded-image duplicates, then uses a reproducible, stratified image-level 80/20 split. Run from the project directory:

```sh
python -m dataset.classifier
python -m unittest discover -s tests
```

The generated `model/boat_svm.xml` and `model/evaluation.json` contain the model, split and predictions. The generated directory is ignored by Git and can be recreated. Label 0 means background; label 1 means boat.

After running the classifier, open `model/review.html` in a browser to see the test images with their actual and predicted labels. Incorrect predictions appear first with a red border. The page contains copies of the test images and is generated locally.

With the images present on 2 November 2026 and seed 42, there were 126 unique images after removing one duplicate. On the 25-image test split, the classifier identified 12 of 12 boat images and 11 of 13 background images; two background images were predicted as boats. These numbers are exploratory: the image source recordings are unknown, and similar frames may occur in both splits. They do **not** measure performance on unseen video.

The classifier is not connected to the video player and does not detect people or fishing. Next, record the source video for each example, split by recording, collect and label representative motion crops from additional clips, and evaluate on a separate recording before integrating predictions into playback. The motion-first approach also misses objects that stop moving.
