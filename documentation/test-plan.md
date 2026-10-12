# Test plan and release evidence

Automated tests are runnable with `python -m pytest -q`. They exercise explicit
shadow removal at threshold 50, class-change ID continuity, one-to-one matching,
expiry, stationary re-inference, no inference on an empty gate, zone validation,
temporal confirmation/reset, negative classes, one snapshot/CSV row, event
duration, direction and duplicate boat crossings, decode/encode and full video
processing, source-group leakage, class completeness, evaluation metrics,
rod-candidate geometry and ground-plane calibration.

The video-to-event test uses a deterministic injected detector to exercise the
event path. It **does not** demonstrate learned fishing recognition. Separate
manual integration checks executed an actual YOLO11n model on its reference
image and through `main.py`. A one-epoch training smoke check on nine synthetic
images produced a three-class checkpoint and held-out metric output; precision,
recall and mAP were zero. This verifies the training machinery, not a useful
fishing model. Those synthetic weights are outside Git and must not be used for
the project demonstration as a trained fishing detector.

## Real recording checked in this branch

The full `person_on_dock_1.mp4` recording was decoded and processed with actual
YOLO11n plus the rod heuristic: 5,769 frames, approximately 96.13 source seconds,
339 model crop calls, zero boat crossings, and **three false fishing candidates**.
The saved confirmation snapshots were reviewed: the person was walking without
a visible rod and the reported line geometry came from the background. This
is evidence that the simple geometry fallback is not sufficiently specific.
It is not a fishing recall measurement or a full frame-labelled benchmark.
The CSV preserves the tentative `fishing_candidate` type. The candidate code
must not be presented as solved fishing recognition.

The GUI pipeline also completed a 38-frame synthetic MP4 under Xvfb with real
OpenCV playback, live parameter trackbars and the event window. Human interaction
with mouse-based zone/annotation setup still needs an accessible desktop.

## Real-clip acceptance work

1. Prepare independent labelled train/val/test recordings with all three classes.
2. Run normal walking, standing, fishing and stop/resume episodes. Compare IDs,
   class labels and event intervals to manually reviewed ground truth.
3. Include no rod, boat hook, long bag, overlapping people, distant people,
   reflections, wet conditions, dusk and an empty-dock hour.
4. Fix a boat counting line/direction, manually count actual crossings and check
   count error. Moored boats that never cross must not count.
5. Evaluate fishing precision/recall, interval matches and false events per hour.
   Keep candidate heuristic results separate from trained-model results.
6. Verify real desktop zone clicking, parameter tuning, pause/resume and event
   inspection. Xvfb checks rendering but not ease of interaction for an operator.
7. If using calibration, measure real ground correspondences and check held-out
   known ground distances. Never infer body height with a plane homography.

The recording-specific accuracy criteria remain unverified until these labelled
checks are performed. Implementation is not an acceptance-test pass.
