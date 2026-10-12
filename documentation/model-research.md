# Model and dataset research — 12 October 2026

## Verified usable starting model

[Ultralytics YOLO11n official weights](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)
were downloaded through verified HTTPS and executed with the adapter. The COCO
model found four people in the official
[bus reference image](https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg).
It includes person/boat, **not person_fishing**. The reference-image check does
not measure dock-camera performance. Review Ultralytics/COCO model and data
licensing before redistribution; weights are retained outside Git.

## Candidate sources, not established fishing models

- [Roboflow fishing recognition](https://universe.roboflow.com/lalaland/fishing-recognition-gv5jm):
  referenced by the original presentation. Proxy access denied during the
  research request. Class definitions, licence, dataset availability and model
  weights were not verified. Do not report it as downloaded or usable.
- [Roboflow boat detector](https://universe.roboflow.com/aqhere/boat-detect-hsxmm):
  likewise not accessible during the research request. Boat-only data cannot
  supply fishing supervision.
- [Fishing detection dataset description on GitHub](https://github.com/767172261/Fishing-recognition-based-on-deep-learning-AI-image-recognition-data-set-Fishing-image-data-set-yolo):
  README retrieved from `main`. It describes 758 images with fishing/no-fishing
  labels. A usable downloadable dataset, compatible person bounding-box labels,
  checkpoint and clear licence were not verified. Description claims are not
  evidence of a trained model suitable for this camera.
- [Keras Fishing Recognition](https://github.com/danielshaving/Keras_Fishing_Recognition):
  README retrieved from `master`. This is the Nature Conservancy fisheries
  monitoring competition (fish-category imagery), not a model for people fishing
  from a dock. Do not use it as a substitute.

GitHub API and Hugging Face model-search access were also denied by the runtime
proxy. Additions for `universe.roboflow.com`, `api.github.com` and
`huggingface.co` were saved in the environment draft. Saving is not proof of
runtime access, publication or credentials. No secret values were requested.

## Project footage

The public [shared Drive folder](https://drive.google.com/drive/folders/1Uu5GTa6J6cQnodDX9FfMKCCXSiRi4DZO)
became accessible during this work. It lists seven boat-named files (one names
boats 6 and 7), two person-on-dock clips, one person-fishing clip and an empty
clip. These filenames are not frame annotations or verified event counts.
The downloaded person/empty clips are 1280×720 at approximately 60 FPS, so the
presentation must not state that every recording is 1920×1080. The stored zone
coordinates need explicit scaling or reselection.

Recommendation: use general pretrained person/boat detection immediately;
test the optional rod heuristic as **possible fishing**; collect and annotate
independent positive and negative sessions before claiming fishing recognition.
No publicly available fishing checkpoint with verified access, compatible labels
and suitable licensing was established in this search. This does not prove that
none exists.
