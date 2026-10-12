# Data, labels and evaluation

The project uses three detector classes in this order: `boat`, `person`,
`person_fishing`. The fishing class is a person engaged in fishing, not a rod
alone. Annotate the person's body box. Walking, standing without a rod, a long
bag and a boat hook are important negatives. A rod heuristic produces
`person_fishing_candidate`, which is not a learned fishing label.

## Manifest

Paths are relative to the manifest or absolute. Boxes are image-pixel xywh.
Only reviewed frames appear in `frames`; `objects: []` explicitly means reviewed
negative. An unlisted frame is unreviewed, not a negative. Combine per-clip JSON
files from `annotate.py` into a single `clips` array.

```json
{
  "clips": [{
    "id": "dock_test_1",
    "group": "recording_session_3",
    "split": "test",
    "video": "media/test.mp4",
    "recorded_at": "2026-10-12",
    "weather": "daylight, dry",
    "empty_dock": false,
    "manual_boat_count": 1,
    "frames": [
      {"index": 120, "objects": [{"label": "person_fishing", "box": [50, 390, 20, 55]}]},
      {"index": 180, "objects": []}
    ],
    "events": [{"type": "fishing", "start_seconds": 2.0, "end_seconds": 10.0}]
  }]
}
```

This is a schema example, not ground truth. Train/val/test each need all three
classes for the dataset builder. The same video or source group cannot span
splits. Different clips cut from one source/session must share `group` so they
cannot leak across splits. Do not split neighbouring frames randomly. The tool
cannot infer related source sessions from filenames; group them honestly.

The available shared folder contains one file named `person_fishing_1.mp4`.
That is not itself a labelled dataset or three independent fishing recordings.
Collect additional independent positive sessions and the specified negatives.
Use training to fit weights, validation to tune confidence, temporal thresholds
and geometric fallback, then reserve test recordings for final evaluation.

## Outputs and metrics

`train_model.py` checks data/provenance, fine-tunes local pretrained YOLO weights,
saves `weights/best.pt`, then validates on the held-out test split and writes
`test_metrics.json` with per-class precision, recall, mAP50 and mAP50–95.
These are Ultralytics model metrics, not fishing-event metrics. Do not repeatedly
tune on the test split.

`evaluate.py` matches run tracks to reviewed frame boxes at IoU ≥ 0.5, one-to-one
and same class, and reports TP/FP/FN, precision and recall. Unobserved retained
tracks still count as predictions. Rod candidates remain `person` for semantic
box metrics; no fishing classification accuracy is credited to the heuristic.
Review candidate events separately using ground-truth type `fishing_candidate`
if measuring agreement of the candidate mechanism. To establish fishing-event
accuracy, annotate actual fishing and compare the candidate flags manually.

If `events` is supplied, fishing intervals match at temporal IoU ≥ 0.5, and boat
crossings match within one second. Event labels must match. `manual_boat_count`
allows count error reporting. Undefined precision/recall is `null`, not a fake
100% or 0% success.

`empty_dock: true` is a human assertion about the entire processed recording.
The evaluator reports confirmed false events per hour and false track-frames
per hour; these are different units. Duration comes from processed source
timestamps, with a frame/FPS fallback. Short clips do not establish hour-long performance. Missing reviewed frames,
wrong source video and motion-only semantic evaluation fail explicitly.

Ground truth needs a second member's review. Log adverse light, reflections,
rain, two overlapping people, distant people, stationary fishing, hooks, long
bags and empty scenes. Obtain per-condition results before claiming robustness.
