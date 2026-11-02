# DockEye

DockEye uses motion detection on recorded video. It has separate zones for the dock and the water. The box colours show which zone detected motion; they don't identify the object yet.

## Setup and playback

Use Python 3.10 or newer. From the project folder, run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Put the videos in `media/`, or pass their paths to `main.py`. The supported video formats in `media/` are ignored by Git. Zone settings are saved in `config/MDZones.json`.

## Boat classifier experiment

This experiment trains an OpenCV linear SVM to tell boat images from background images. It resizes the images in `dataset/boat/` and `dataset/background/` to 32×32 pixels and converts them to grayscale. Exact duplicate images are removed before an 80/20 train/test split.

To train it and run the tests:

```sh
python -m dataset.classifier
python -m unittest discover -s tests
```

The script saves the model to `model/boat_svm.xml` and the test predictions to `model/evaluation.json`. The `model/` folder is ignored by Git; run the script again to recreate it. In the results, 0 means background and 1 means boat.

Open `model/review.html` in a browser to see the test images alongside their labels and predictions. Mistakes appear first with a red border.

With the current dataset and seed 42, the script found 126 unique images and removed one duplicate. On the 25-image test split, it got 12 of 12 boat images and 11 of 13 background images right. We don't know which recordings the screenshots came from, so similar frames may be in both the training and test sets. These results don't tell us how well the model works on a new recording.

The classifier runs separately from the video player. It doesn't detect people or fishing. To evaluate it properly, we need labelled crops from known recordings and a separate recording for testing. Motion detection can also miss an object that stops moving.
