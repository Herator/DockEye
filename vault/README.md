# Vault

Old code we don't use anymore but didn't want to throw away. The folders match the main project, so `vault/src/fishing.py` used to be `src/fishing.py`.

- `src/fishing.py`, `dataset/fishing_crops.py`, `dataset/train_fishing.py`, `tests/test_fishing.py`: the fishing classifier. We dropped it because we only had one clip of someone fishing, which isn't enough to train it and test it on other clips.
- `src/tracker.py`: our first tracker, which kept a box on a person after they stopped moving. YOLO with ByteTrack does that now.

To use something again, move it back to the same place in the project. For fishing, also uncomment the lines marked `fishing dropped` in `main.py` and `src/events.py`.
