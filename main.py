import sys
from concurrent.futures import ThreadPoolExecutor

import cv2

from src.motionDetection import motion_detection, boat_backSub, person_backSub, person_kernel
from src.zones import make_zones, draw_the_zone
from src.mediaPlayer import Player, collect
from src.detector import Detector
from src.events import EventLog
# from src.fishing import FishingClassifier  # fishing dropped, see vault/

# -------------------------------------
#             SETTINGS
# -------------------------------------
MEDIA = "media/"
MD_ZONES_FILE = "config/MDZones.json"
MD_COLORS = {"person": (0, 0, 255), "boat": (255, 0, 0)}  # zone outlines
COLORS = {"person": (0, 165, 255), "boat": (0, 200, 0)}    # boxes (BGR)
# COLORS["person_fishing"] = (0, 0, 255)  # fishing dropped

MOTION_SCALE = 0.5  # motion only wakes us up, half size is enough
MOTION_EVERY = 2    # every 2nd frame is enough too
WARMUP_S = 3        # time for MOG2 to learn the empty scene
IDLE_S = 5          # sleep after this long with nothing going on
YOLO_EVERY_S = 0.2  # while awake. Slower and distant boats lose their ID
CHECK_S = 10        # while asleep, in case someone is standing still

cv2.setNumThreads(4)  # all cores was slower while YOLO runs

# -------------------------------------
#               MEDIA
# -------------------------------------
paths = collect(sys.argv[1:] or [MEDIA])
if not paths:
    raise SystemExit(f"No videos found in {MEDIA!r}")

cam = cv2.VideoCapture(paths[0])
ret, first = cam.read()

# skip black frames at the start
while ret and first.mean() < 10:
    ret, first = cam.read()
cam.release()
if not ret:
    raise SystemExit(f"{paths[0]} has no non-black frames")

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

# ------------------------------------
#                ZONES
# ------------------------------------
md_zones = make_zones(first, MD_COLORS, MD_ZONES_FILE)
cv2.destroyWindow("Video")
small_zones = {k: [[int(x * MOTION_SCALE), int(y * MOTION_SCALE)] for x, y in pts] for k, pts in md_zones.items()}
small_boat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # half of boat_kernel

# ------------------------------------
#          PER-FRAME DETECTION
# ------------------------------------
log = EventLog()
yolo = ThreadPoolExecutor(max_workers=1)  # background, so the video never waits for YOLO
state = {}

# fishing dropped: to bring it back, uncomment this and submit run_yolo further down.
# fishing = FishingClassifier()
#
# def run_yolo(detector, frame):
#     tracks = detector.track(frame)
#     tracks["person_fishing"] = {i: b for i, b in tracks["person"].items() if fishing.is_fishing(frame, b)}
#     return tracks


def detect(frame):
    video = player.paths[player.i]
    if video != log.video:  # new video, start over
        if state.get("job"):
            state["job"].result()
        log.start_video(video)
        state.update(detector=Detector(md_zones), awake=True, last_active=0.0,
                     last_yolo=float("-inf"), tracks={}, job=None)
        # learningRate=1 forgets the last clip's background
        small = cv2.resize(frame, None, fx=MOTION_SCALE, fy=MOTION_SCALE, interpolation=cv2.INTER_AREA)
        person_backSub.apply(small, learningRate=1)
        boat_backSub.apply(cv2.GaussianBlur(small, (21, 21), 0), learningRate=1)

    t = player.pos / player.fps
    if t < state["last_yolo"]:  # jumped backwards
        state["last_yolo"] = float("-inf")

    if player.pos % MOTION_EVERY == 0:
        small = cv2.resize(frame, None, fx=MOTION_SCALE, fy=MOTION_SCALE, interpolation=cv2.INTER_AREA)
        a = MOTION_SCALE ** 2  # min areas shrink with the frame
        person_boxes = motion_detection(small, small_zones["person"], 350 * a, 150, person_backSub, person_kernel)
        boat_boxes = motion_detection(small, small_zones["boat"], 1000 * a, 50, boat_backSub, small_boat_kernel, blur=True)
        if t >= WARMUP_S and (person_boxes or boat_boxes):
            state.update(awake=True, last_active=t)

    # YOLO done? Its boxes are a few frames old, that's fine
    job = state["job"]
    if job is not None and job.done():
        tracks, yolo_t, yolo_frame = job.result(), state["job_t"], state["job_frame"]
        state["job"] = None
        if tracks["person"] or tracks["boat"]:  # someone standing still keeps us awake
            state.update(awake=True, last_active=max(state["last_active"], yolo_t))
        for cls, cls_tracks in tracks.items():
            log.update(cls, cls_tracks, yolo_frame, yolo_t)
        state["tracks"] = tracks

    every = YOLO_EVERY_S if state["awake"] else CHECK_S
    if state["job"] is None and t - state["last_yolo"] >= every:
        state.update(last_yolo=t, job_t=t, job_frame=frame.copy())  # copy, we draw on frame below
        state["job"] = yolo.submit(state["detector"].track, state["job_frame"])
        # state["job"] = yolo.submit(run_yolo, state["detector"], state["job_frame"])  # fishing dropped

    if state["awake"] and t - state["last_active"] > IDLE_S:
        state.update(awake=False, tracks={})
        for cls in COLORS:
            log.update(cls, {}, frame, t)  # end open events

    for cls, cls_tracks in state["tracks"].items():
        for tid, (x, y, w, h) in cls_tracks.items():
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS[cls], 2)
            cv2.putText(frame, f"{cls} {tid}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[cls], 2)
    status = "AWAKE" if state["awake"] else "SLEEPING"
    cv2.putText(frame, f"{status}  Boats: {log.count.get('boat', 0)}  People: {log.count.get('person', 0)}",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # fishing dropped, the status line used to end with:  Fishing: {log.count.get('person_fishing', 0)}

    # zones last, or the lines would count as motion
    draw_the_zone(frame, md_zones, MD_COLORS)
    return frame


player = Player(paths, detect)
player.toggles["boxes"] = True  # detection on from the start ('o' toggles)
player.run()
if state.get("job"):
    state["job"].result()
yolo.shutdown()
log.close()  # write events still open
