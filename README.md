# DockEye

DockEye finds people on the dock and boats passing it in recorded video, and logs when they were there. Motion detection, with separate zones for the dock and the water, decides when to look. A pretrained YOLO model decides what is there.

## Setup and playback

Use Python 3.10 or newer. From the project folder, run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Put the videos in `media/`, or pass their paths to `main.py`. The supported video formats in `media/` are ignored by Git. Zone settings are saved in `config/MDZones.json`.

## How it works

### Sleep and wake

Motion detection (MOG2) is only there to wake the program up. It runs on a half size frame, on every second frame. When nothing has moved for 5 seconds DockEye goes to sleep, and then YOLO only runs every 10 seconds in case someone is standing still. Motion wakes it up again and YOLO goes back to 5 runs a second. Motion is ignored for the first 3 seconds of each video while MOG2 learns what the empty scene looks like. You can see if it's awake or sleeping in the bottom left corner.

### Detection and tracking

YOLO decides what something is, the zones only decide where it is. We use the pretrained `yolo26s` model at 1280 px. The smaller `n` model missed people at the far end of the dock. YOLO only gets the rectangle around the zones, so it doesn't waste time on the hills. The model is downloaded to `model/` the first time.

A person only counts if their feet are inside the dock zone. That means someone sitting in a boat next to the dock isn't counted. A boat only counts after it has moved 40 px, otherwise all the moored boats would show up all the time. The downside is that a boat that is already parked when the video starts is never shown.

ByteTrack (`config/tracker.yaml`) gives every object an ID. Boats far away are only about 35x12 px, and with the default settings they never kept an ID, so we changed how new tracks are confirmed (`DockTracker` in `src/detector.py`). That's also why YOLO runs 5 times a second. With fewer runs a distant boat moves more than its own width between them and gets a new ID.

### Events

Anything seen for at least 2 seconds is an event. We save one snapshot to `output/snapshots/`, and when the object has been gone for 1 second a row goes into `output/events.csv` with video, class, ID, event, start, end, duration and snapshot. The counts on screen are the events in the current video.

### Speed

YOLO runs through ONNX Runtime, which was about twice as fast as PyTorch on our laptops. The first time you run it, `model/yolo26s.pt` is exported to an ONNX file that fits your zones. That takes about 20 seconds. One YOLO run still takes around 0.2 s, so it runs in a background thread and the video uses the last result it got. It only gets a quarter of the CPU cores, since the video slowed down a lot when it used all of them. On our laptop a 30 fps clip plays at about 24-28 fps while awake.

## Fishing detection (dropped)

The plan was to tell people fishing apart from people just standing on the dock. We wrote a classifier for it, but we only had one clip with someone fishing. That isn't enough to train it and then test it on clips it hasn't seen, so we couldn't make it reliable and took it out. The online datasets we found didn't help much either. They are mostly close-up photos, and our camera sees people far away and from the side. To get it working you would need more of your own clips with people fishing, plus negatives like someone carrying a boat hook or a long bag.

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

The classifier runs separately from the video player and isn't used by `main.py`. To evaluate it properly, we need labelled crops from known recordings and a separate recording for testing. Motion detection can also miss an object that stops moving.
