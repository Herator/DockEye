# DockEye

DockEye finds people on the dock and boats passing it in recorded video, and logs when they were there. Motion detection, with separate zones for the dock and the water, decides when to look. A pretrained YOLO model decides what is there.

## Installation

You need Python 3.10 or newer and Git. The first run also needs internet, to download the YOLO model.

1. Clone the repo and go into it:

   ```sh
   git clone <repo url>
   cd dockeye
   ```

2. Make a virtual environment and install the packages. This pulls in PyTorch, so it can take a few minutes.

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

   macOS / Linux:
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. Get the videos. They are too big for GitHub, so they're on our [Google Drive](https://drive.google.com/drive/folders/1Uu5GTa6J6cQnodDX9FfMKCCXSiRi4DZO?usp=sharing) (the link is also in `media/videos_link.txt`). Download them into the `media/` folder. Git ignores video files there.

4. Run it:

   ```sh
   python main.py
   ```

   It starts on the webcam. Press `v` to get the list and click a video to play it instead, or "Live camera 0" at the top to go back. To play just one video, give its path: `python main.py media/boat_1.mp4`.

## First run

The first time, it downloads `yolo26s.pt` to `model/` and converts it to ONNX, which takes about 20 seconds. After that it starts straight away.

The dock and water zones for our camera are already saved in `config/MDZones.json`. If you use your own footage from another camera, delete that file. On the next run you get asked to click the corners of the dock zone and then the water zone, pressing Enter after each one.

Events end up in `output/events.csv` and snapshots in `output/snapshots/`.

## Live camera

The camera has its own zones in `config/MDZones_live.json`, since it doesn't see the same thing as our dock videos. The first time you open it you click them like above, and after that they're loaded.

While it runs it keeps the last 2 minutes in memory (`BUFFER_S` in `src/mediaPlayer.py`). So if you miss something you can pause or drag the slider back and watch it again, and it keeps recording in the meantime. The Live button turns blue when you're behind, click it (or press End) to get back. Detection only runs on what you're watching though, so anything that happens live while you're looking back isn't logged.

Events from the camera go in the same `output/events.csv`. The video column is the time you opened the camera, like `live0_20261104-112955`, and start/end are seconds after that.

An IP camera should work too if you give it the stream URL, `python main.py rtsp://user:pass@192.168.1.20/stream`. We have only tried a webcam.

## Controls

| Key | What it does |
|---|---|
| Space | Play / pause |
| Left / Right arrow | One frame back / forward |
| `j` / `l` | 10 seconds back / forward |
| End | Back to live (camera) / last frame (video) |
| `p` / `n` | Previous / next video |
| `-` / `+` | Slower / faster |
| `v` | Show the video list (click one to play it) |
| `o` | Detection on/off |
| `i` | Info text on/off |
| `r` | Loop the video |
| `f` | Flip the image |
| `q` or Esc | Quit |

All of these are also buttons under the video.

### Adjusting boat motion detection

The `Boat tuning` window opens with the player. Use `Minimum area` to change the smallest motion region that gets a box. `Mode 1` keeps only grey pixels from the MOG2 mask (the shadow-only experiment); `Mode 0` uses the threshold slider. In mode 0, a threshold of 50 keeps grey and white pixels, while 150 keeps white pixels only. The threshold slider has no effect in mode 1.

The two mask windows show the boat mask before and after filtering, before the water zone is applied. Pause the video with Space (the slider changes then redraw the same frame) to compare settings on the same frame. Settings are temporary and reset when the program closes. When you switch videos, seek, or loop back to the start, the background model resets. A message appears while it learns from approximately three seconds of processed frames; detections can be unreliable during that time and may take longer to settle.

## Running the tests

From the project folder, with the virtual environment active:

```sh
python -m unittest discover -s tests
```

All 13 tests should pass.

## If something goes wrong

- `module 'cv2' has no attribute 'ml'`: you have OpenCV 5, which doesn't have the part the boat classifier needs. Run `python -m pip install "opencv-python>=4.10,<5"`. A fresh install from `requirements.txt` already gets version 4.
- `No videos found in 'media/'`: the videos aren't downloaded yet, see step 3.
- PowerShell won't run `Activate.ps1`: run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.
- The video plays slower than normal: that's YOLO using the CPU. It runs at about 24-28 fps on our laptops while something is happening. Close other heavy programs, or press `o` to turn detection off.

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

The fishing code is kept in `vault/` together with our first tracker, and the lines that used it in `main.py` are commented out. See `vault/README.md` for how to bring it back.

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
