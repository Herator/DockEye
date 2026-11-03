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

### Adjusting boat motion detection

The `Boat tuning` window opens with the player. Use `Minimum area` to change the smallest motion region that gets a box. `Mode 1` keeps only grey pixels from the MOG2 mask (the shadow-only experiment); `Mode 0` uses the threshold slider. In mode 0, a threshold of 50 keeps grey and white pixels, while 150 keeps white pixels only. The threshold slider has no effect in mode 1.

The two mask windows show the boat mask before and after filtering, before the water zone is applied. Pause the video with Space to compare settings on the same frame. Settings are temporary and reset when the program closes. When you switch videos, seek, or loop back to the start, the background model resets. A message appears while it learns from approximately three seconds of processed frames; detections can be unreliable during that time and may take longer to settle.

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

## YOLO video experiment

This separate experiment runs pretrained YOLO11n on whole video frames. It draws boxes for the COCO `person` and `boat` classes; it has not been trained on our footage and does not recognise fishing. The existing player and motion detector do not use this model.

Install the optional dependencies and download the official model weights into the ignored `model/` folder:

```sh
python -m pip install -r requirements-yolo.txt
mkdir -p model
curl -fL -o model/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

To process every frame of a video and save the result:

```sh
python experiments/yolo_video.py media/boat_3.mp4 --output model/boat_3_yolo.mp4 --confidence 0.25
python src/mediaPlayer.py model/boat_3_yolo.mp4
```

The script also saves a JSON file with the model's boxes and scores for each frame. A score is the model's confidence, not measured accuracy. Use a new output name when changing the confidence threshold: the script will not overwrite an existing video or JSON file. If the decoded frame count differs from the video's reported frame count, the script prints a warning; check the source video and output before assuming the full clip was processed.

To inspect a few frames from several clips instead:

```sh
python experiments/yolo_samples.py media/boat_3.mp4 media/boat_8.mp4 media/bølger_1.mp4
```

Open `model/yolo_samples/review.html` to see the results. Use `--output-dir` to choose a new folder for another run. The generated videos, images, predictions, and weights remain local and are ignored by Git. The Ultralytics package has its own [licensing terms](https://www.ultralytics.com/license).
