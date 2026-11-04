# Video and live camera player with a toggle that runs detection on each frame.

import glob
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

# used when no paths are given on the command line
VIDEO_PATH = "../media/"

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
WIN = "Player"
W = 960          # display width of the video area
ROW_H = 36       # height of each button row
BAR_H = ROW_H * 2 + 10
LIST_TOP, LIST_ROW, LIST_W = 40, 26, 420   # video list overlay layout
BUFFER_S = 120   # how far back you can go on a live camera. JPEGs in memory, roughly 1-3 MB per second


def collect(args):
    paths = []
    for a in args:
        if os.path.isdir(a):
            for ext in VIDEO_EXTS:
                paths += glob.glob(os.path.join(a, "*" + ext))
        else:
            paths.append(a)
    return sorted(paths)


def capture(path):
    """A video file, a stream URL (rtsp://...) or a camera number like '0'."""
    return cv2.VideoCapture(int(path) if path.isdigit() else path)


def display_name(path):
    return f"Live camera {path}" if path.isdigit() else os.path.basename(path)


class Player:
    TRANSPORT = [
        ("|<", "prev"), ("<<10s", "back10"), ("<-", "stepb"), ("Play/Pause", "play"),
        ("->", "stepf"), ("10s>>", "fwd10"), ("Live", "live"), (">|", "next"), ("Slower", "slower"), ("Faster", "faster"),
    ]
    TOGGLES = [("Videos", "list"), ("Info", "info"), ("Flip", "flip"),
               ("Boxes", "boxes"), ("Loop", "loop")]

    def __init__(self, paths, boxes=lambda frame: frame):
        self.paths = paths
        self.boxes = boxes
        self.cache_key, self.cache = None, None
        self.cap = None
        self.thread = None
        self.paused = False
        self.speed = 1.0
        self.toggles = {"list": False, "info": True, "flip": False,
                        "boxes": False, "loop": False}
        self.list_scroll = 0
        self.video_h = 540
        self.seek_req = None
        self.updating_bar = False
        self.quit = False
        self.open(0)

    # ---------- video handling ----------
    def open(self, i):
        self.close()
        self.i = i % len(self.paths)
        self.cap = capture(self.paths[self.i])
        count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.live = count <= 0  # cameras and streams have no length
        self.n = 0 if self.live else int(count)  # live: frames recorded so far
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.t0 = self.frame_t = time.perf_counter()
        self.frame = None
        self.pos = -1
        if self.live:
            self.buf = deque(maxlen=int(BUFFER_S * self.fps))  # (time, jpeg)
            self.latest, self.ended, self.recording = None, False, True
            self.new = threading.Condition()
            self.thread = threading.Thread(target=self.record, daemon=True)
            self.thread.start()
        if not self.next_frame():  # no camera, or a broken file
            self.frame = np.zeros((540, W, 3), np.uint8)
            cv2.putText(self.frame, f"Could not open {display_name(self.paths[self.i])}", (20, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    def close(self):
        if self.thread is not None:
            self.recording = False
            self.thread.join(timeout=2)
            self.thread = None
        if self.cap is not None:
            self.cap.release()

    def record(self):
        """Live: reads the camera all the time, also while paused or watching the past."""
        while self.recording:
            ok, f = self.cap.read()
            if not ok:
                break
            jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
            with self.new:
                self.buf.append((time.perf_counter(), jpg))
                self.latest = f
                self.n += 1
                if len(self.buf) > 30:  # cameras often deliver fewer fps than they report
                    self.fps = (len(self.buf) - 1) / (self.buf[-1][0] - self.buf[0][0])
                self.new.notify()
        with self.new:
            self.ended = True
            self.new.notify()

    def bounds(self):
        """First and last frame you can go to."""
        return (self.n - len(self.buf), self.n - 1) if self.live else (0, self.n - 1)

    def at_live_edge(self):
        return self.live and self.pos >= self.n - 1

    def show(self, pos, wait=False):
        """Live: show frame pos from the buffer. wait: block until the camera has it."""
        with self.new:
            if wait:
                self.new.wait_for(lambda: self.n > pos or self.ended, timeout=5)
            first, last = self.bounds()
            pos = max(first, min(last, int(pos)))
            if last < 0 or (wait and pos == self.pos):
                return False  # nothing new from the camera
            t, jpg = self.buf[pos - first]
            latest = self.latest if pos == last else None  # live: skip the decode and the JPEG blur
        self.frame = latest if latest is not None else cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        self.pos, self.frame_t = pos, t
        return True

    @property
    def t(self):
        """Seconds into the video, or since the camera was opened."""
        return self.frame_t - self.t0 if self.live else self.pos / self.fps

    def next_frame(self):
        if self.live:
            return self.show(self.pos + 1, wait=True)
        ok, f = self.cap.read()
        if ok:
            self.frame = f
            self.pos += 1
        return ok

    def seek(self, pos):
        if self.live:
            return self.show(pos)
        pos = max(0, min(self.n - 1, int(pos)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, f = self.cap.read()
        if ok:
            self.frame = f
            self.pos = pos
        return ok

    # ---------- actions ----------
    def do(self, name):
        if name == "prev":
            self.open(self.i - 1)
        elif name == "next":
            self.open(self.i + 1)
        elif name == "back10":
            self.seek(self.pos - self.fps * 10)
        elif name == "fwd10":
            self.seek(self.pos + self.fps * 10)
        elif name == "stepb":
            self.paused = True
            self.seek(self.pos - 1)
        elif name == "stepf":
            self.paused = True
            self.next_frame()
        elif name == "live":
            self.seek(self.n - 1)  # newest frame, or the last one in a video
        elif name == "play":
            self.paused = not self.paused
        elif name == "slower":
            self.speed = max(0.25, self.speed / 2)
        elif name == "faster":
            self.speed = min(4.0, self.speed * 2)
        elif name in self.toggles:
            self.toggles[name] = not self.toggles[name]

    # ---------- UI ----------
    def buttons(self):
        """Return list of (rect, label, action) with rects relative to the control bar."""
        out = []
        for row, items in enumerate((self.TRANSPORT, self.TOGGLES)):
            w = W // len(items)
            for k, (label, action) in enumerate(items):
                x0, y0 = k * w + 2, row * ROW_H + 5
                out.append(((x0, y0, x0 + w - 4, y0 + ROW_H - 4), label, action))
        return out

    def list_rows(self):
        return max(1, (self.video_h - LIST_TOP - 10) // LIST_ROW)

    def on_mouse(self, event, x, y, flags, param):
        in_video = y < self.video_h
        # scroll the video list with the mouse wheel
        if event == cv2.EVENT_MOUSEWHEEL and in_video and self.toggles["list"]:
            step = -1 if flags > 0 else 1  # same sign as the wheel delta, getMouseWheelDelta is gone in OpenCV 5
            max_scroll = max(0, len(self.paths) - self.list_rows())
            self.list_scroll = max(0, min(max_scroll, self.list_scroll + step))
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # click on a video in the list
        if in_video:
            if self.toggles["list"] and x < LIST_W and y >= LIST_TOP:
                idx = self.list_scroll + (y - LIST_TOP) // LIST_ROW
                if 0 <= idx < len(self.paths):
                    self.open(idx)
            return
        ly = y - self.video_h
        for (x0, y0, x1, y1), _, action in self.buttons():
            if x0 <= x <= x1 and y0 <= ly <= y1:
                self.do(action)
                return

    def on_trackbar(self, val):
        if not self.updating_bar:
            self.seek_req = val

    def draw_bar(self):
        bar = np.full((BAR_H, W, 3), 35, np.uint8)
        for (x0, y0, x1, y1), label, action in self.buttons():
            if action == "play":
                label = "Play" if self.paused else "Pause"
            on = self.toggles.get(action, False)
            if action == "live":
                label = "Live" if self.live else "End"
                on = self.live and not self.at_live_edge()  # lit while you're behind live
            color = (200, 120, 40) if on else (80, 80, 80)
            cv2.rectangle(bar, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(bar, (x0, y0), (x1, y1), (150, 150, 150), 1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(bar, label, ((x0 + x1 - tw) // 2, (y0 + y1 + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return bar

    def render(self):
        # detect once per frame, pausing shouldn't feed MOG2 the same frame
        key = (self.i, self.pos, self.toggles["flip"], self.toggles["boxes"])
        if key != self.cache_key:
            f = self.frame.copy()
            if self.toggles["flip"]:
                f = cv2.flip(f, 1)
            if self.toggles["boxes"]:
                f = self.boxes(f)
            self.cache_key, self.cache = key, f
        f = self.cache
        h = int(f.shape[0] * W / f.shape[1])
        f = cv2.resize(f, (W, h))
        self.video_h = h
        if self.toggles["info"]:
            if self.at_live_edge():
                where = f"LIVE  {self.t:.0f}s"
            elif self.live:
                where = f"LIVE -{time.perf_counter() - self.frame_t:.1f}s (recording)  x{self.speed:g}"
            else:
                where = f"frame {self.pos + 1}/{self.n}  {self.t:.1f}s  x{self.speed:g}"
            txt = (f"{display_name(self.paths[self.i])} "
                   f"[{self.i + 1}/{len(self.paths)}]  {where}  {'PAUSED' if self.paused else 'PLAYING'}")
            cv2.putText(f, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(f, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if self.toggles["list"]:
            self.draw_list(f)
        return np.vstack([f, self.draw_bar()])

    def draw_list(self, f):
        rows = self.list_rows()
        # keep the current video visible
        if self.i < self.list_scroll:
            self.list_scroll = self.i
        elif self.i >= self.list_scroll + rows:
            self.list_scroll = self.i - rows + 1
        self.list_scroll = max(0, min(max(0, len(self.paths) - rows), self.list_scroll))

        roi = f[:, :LIST_W]
        f[:, :LIST_W] = cv2.addWeighted(roi, 0.25, np.zeros_like(roi), 0.75, 0)
        cv2.putText(f, f"Videos ({len(self.paths)}) - click to play, wheel to scroll", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for r in range(rows):
            idx = self.list_scroll + r
            if idx >= len(self.paths):
                break
            y = LIST_TOP + r * LIST_ROW
            if idx == self.i:
                cv2.rectangle(f, (0, y), (LIST_W, y + LIST_ROW - 2), (200, 120, 40), -1)
            label = display_name(self.paths[idx])
            if len(label) > 48:
                label = label[:45] + "..."
            cv2.putText(f, label, (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

    def sync_trackbar(self):
        first, last = self.bounds()
        val = int((self.pos - first) / max(last - first, 1) * 1000)
        self.updating_bar = True
        cv2.setTrackbarPos("Frame", WIN, val)
        self.updating_bar = False

    # ---------- main loop ----------
    def run(self):
        if sys.platform == "win32":
            import ctypes
            ctypes.WinDLL("winmm").timeBeginPeriod(1)  # else waitKey sleeps 15 ms minimum
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        cv2.createTrackbar("Frame", WIN, 0, 1000, self.on_trackbar)

        keymap = {
            # arrow codes from waitKeyEx: Windows (0x250000/0x270000), Linux GTK (65361/65363)
            0x250000: "stepb", 0x270000: "stepf", 65361: "stepb", 65363: "stepf",
            0x230000: "live", 65367: "live",  # End
            ord(" "): "play", ord("j"): "back10",
            ord("l"): "fwd10", ord("p"): "prev", ord("n"): "next", ord("-"): "slower",
            ord("+"): "faster", ord("="): "faster", ord("i"): "info",
            ord("f"): "flip", ord("o"): "boxes", ord("r"): "loop", ord("v"): "list",
        }

        next_t = time.perf_counter()
        while not self.quit:
            if self.seek_req is not None:
                first, last = self.bounds()
                self.seek(first + self.seek_req / 1000 * (last - first))
                self.seek_req = None
            elif not self.paused:
                if not self.next_frame():  # end of video
                    if self.toggles["loop"]:
                        self.seek(0)
                    else:
                        self.paused = True

            cv2.imshow(WIN, self.render())
            self.sync_trackbar()

            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break

            if self.paused:
                wait, next_t = 30, time.perf_counter()
            elif self.at_live_edge():
                wait, next_t = 1, time.perf_counter()  # next_frame() already waits for the camera
            else:
                # running deadline, so if waitKey oversleeps the next frame makes up for it
                frame_time = 1.0 / (self.fps * self.speed)
                next_t = max(next_t + frame_time, time.perf_counter() - frame_time)  # max 1 frame behind
                wait = max(1, int((next_t - time.perf_counter()) * 1000))
            key = cv2.waitKeyEx(wait)  # waitKey drops arrow keys
            if key not in keymap:
                key &= 0xFF
            if key in (ord("q"), 27):
                break
            if key in keymap:
                self.do(keymap[key])

        self.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    paths = collect(sys.argv[1:] or [VIDEO_PATH])
    if not paths:
        sys.exit(f"No videos found in {VIDEO_PATH!r}. Set VIDEO_PATH at the top of the file.")
    Player(paths).run()