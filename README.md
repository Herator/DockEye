# DockEye

Offline analysis of one fixed dock camera: motion proposals, YOLO classification,
object IDs, time-confirmed fishing events, line-crossing boat counts, CSV,
event snapshots, a review window and parameter controls.

**Model code is implemented; fishing accuracy is not yet established.** A general
COCO model recognises people and boats, but has no fishing class. Train custom
weights using `boat`, `person`, `person_fishing`, or try the explicitly tentative
rod heuristic. Neither the synthetic tests nor the training smoke check measures
accuracy on dock footage.

## Install (Python 3.12, Linux CPU)

```bash
uv venv .venv
uv pip install --python .venv/bin/python 'torch==2.14.1+cpu' 'torchvision==0.29.1+cpu' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements-dev.txt -c requirements-cpu.lock
```

Desktop mode needs an accessible display and OpenCV system libraries. On the
prepared cloud machine, use `/workspace/.dockeye/venv/bin/python`. The optional
Xvfb wrapper `/workspace/.dockeye/run.sh` supplies a virtual display for technical
validation, not an interactive browser desktop. Headless mode needs no display.

## Run

Supply a local video; the recordings are linked in `media/videos_link.txt`.
The default `media/Oscar.mp4` is not included in Git. Each run writes a **new**
directory; existing outputs are rejected, not overwritten.

```bash
# Diagnostic motion regions only; no semantic labels or fishing events.
.venv/bin/python main.py --video media/clip.mp4 --motion-only --headless

# Full custom model, with real fishing class required.
.venv/bin/python main.py --video media/clip.mp4 --weights models/best.pt --require-fishing --headless --save-video

# Experimental alternative: person + rod-like line, logged as fishing_candidate.
.venv/bin/python main.py --video media/clip.mp4 --weights models/yolo11n.pt --fishing-method rod --headless
```

Obtain the general YOLO11n weights from the official
[Ultralytics asset release](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt).
Check the upstream software/model licence before distributing derived models.
DockEye never silently downloads a missing runtime model.

`config/MDZones.json` contains polygons for a 1920×1080 view. For the **same
camera view** resized to 1280×720, add `--zone-reference-size 1920 1080`.
This scales the polygons in memory and does not edit the saved file. A different
camera angle needs newly clicked zones: use `--zones runs/my-zones.json` in
desktop mode. Headless mode requires an existing valid zones file.

Without `--headless`, press **q** to stop and **Space** to pause/resume.
The **Parameters** window changes area, mask threshold, MOG2 variance,
closing kernel and blur during playback. **Events** lists confirmed outputs.
Final parameters are recorded in `summary.json`. Processing slower than the
video frame rate slows desktop playback; processing speed never changes event
timestamps, which use decoded source timestamps (with a monotonic frame/FPS
fallback if the decoder supplies no usable timestamp).

Outputs: `events.csv`, one JPEG per event, `predictions.jsonl`, `summary.json`
and optionally `annotated.mp4`. Times are seconds from the original video start,
including skipped black frames. Only confirmed events create snapshots.
An already confirmed event's duration is finalized when evidence ends or the
run finishes. Boat-crossing duration is zero because it is an instantaneous event.

## Train and evaluate

```bash
.venv/bin/python annotate.py --video media/clip.mp4 --clip-id session1 --group recording1 --split train --output session1.json
# Merge the reviewed clips arrays into a single manifest.json; collect separate
# source recordings for train, validation and test, including all three classes.
.venv/bin/python dataset_tool.py manifest.json --output datasets/dock
.venv/bin/python train_model.py --data datasets/dock/dataset.yaml --base-model models/yolo11n.pt --output runs/training --epochs 50 --device cpu
.venv/bin/python main.py --video media/test.mp4 --weights runs/training/weights/best.pt --require-fishing --headless --output runs/test
.venv/bin/python evaluate.py --manifest manifest.json --clip test_clip --run runs/test --output test_report.json
.venv/bin/python -m pytest -q
```

The annotation UI uses **b** (boat), **p** (person), **f** (person fishing),
rectangle selection, **u** (undo), **Enter** (accept reviewed frame, including an
empty frame), and **q** (stop). Annotate a fishing person once as `person_fishing`,
not as overlapping person and fishing boxes. Have a second member check labels.

See [data and evaluation](documentation/data-and-evaluation.md),
[architecture](documentation/architecture.md), [test plan](documentation/test-plan.md),
[model research](documentation/model-research.md) and the
[Norwegian presentation guide](documentation/presentation-guide.md).
See also the [requirements/evidence matrix](documentation/requirements-status.md).

## Limits

- Motion gates new candidates; active tracks continue to be classified when still.
  People already stationary when the background is learned may never trigger.
- IDs use one-to-one overlap/distance association and a short missed-detection
  grace period. Occlusion, close objects and leaving/re-entering can change IDs.
  Boat counting is once per ID crossing a configured line, not guaranteed global
  re-identification. Set the line and direction with `--boat-line`/`--boat-direction`.
- Rod candidates use geometry, not activity understanding. Boat hooks, branches,
  bags and diagonal railings can produce candidates; small or occluded rods can
  be missed. Review these events separately from learned fishing events.
- Waves, reflections, weather and camera movement require evaluation on real
  clips. No measured detection accuracy or empty-dock false-positive rate is
  claimed for the supplied recordings yet.
- Optional `--calibration calibration.json` projects a person's foot position
  onto measured dock-plane coordinates. It does not infer body height, boat size
  or rod angle in 3D; measured correspondence points must be supplied.
