# Media player module
# Option to select what media to play
# Basic buttons like pause, player, forward, backward
# Toggl buttons for zones and boxes

import glob
import os
import sys
import time

import cv2
import numpy as np

# Folder (or list of files) to play. Used when no command-line arguments are given.
VIDEO_PATH = "../media/"

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
WIN = "Player"
W = 960          # display width of the video area
ROW_H = 36       # height of each button row
BAR_H = ROW_H * 2 + 10
LIST_TOP, LIST_ROW, LIST_W = 40, 26, 420   # video list overlay layout


def boxes_frame(frame):
    """Put your own per-frame boxesing here (detection, drawing, etc.).
    It only runs while the 'boxes' toggle is on."""
    return cv2.Canny(frame, 100, 200) if False else frame  # <- replace


def collect(args):
    paths = []
    for a in args:
        if os.path.isdir(a):
            for ext in VIDEO_EXTS:
                paths += glob.glob(os.path.join(a, "*" + ext))
        else:
            paths.append(a)
    return sorted(paths)


class Player:
    TRANSPORT = [
        ("|<", "prev"), ("<<10s", "back10"), ("<", "stepb"), ("Play/Pause", "play"),
        (">", "stepf"), ("10s>>", "fwd10"), (">|", "next"), ("Slower", "slower"), ("Faster", "faster"),
    ]
    TOGGLES = [("Videos", "list"), ("Info", "info"), ("Flip", "flip"),
               ("Boxes", "boxes"), ("Loop", "loop")]

    def __init__(self, paths, boxes=boxes_frame):
        self.paths = paths
        self.boxes = boxes
        self.cache_key, self.cache = None, None
        self.cap = None
        self.paused = False
        self.speed = 1.0
        self.toggles = {"list": len(paths) > 1, "info": True, "flip": False,
                        "boxes": False, "loop": False}
        self.list_scroll = 0
        self.video_h = 540
        self.seek_req = None
        self.updating_bar = False
        self.quit = False
        self.open(0)

    # ---------- video handling ----------
    def open(self, i):
        if self.cap is not None:
            self.cap.release()
        self.i = i % len(self.paths)
        self.cap = cv2.VideoCapture(self.paths[self.i])
        self.n = max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame = None
        self.pos = -1
        self.next_frame()

    def next_frame(self):
        ok, f = self.cap.read()
        if ok:
            self.frame = f
            self.pos += 1
        return ok

    def seek(self, pos):
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
            step = -1 if cv2.getMouseWheelDelta(flags) > 0 else 1
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
            color = (200, 120, 40) if on else (80, 80, 80)
            cv2.rectangle(bar, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(bar, (x0, y0), (x1, y1), (150, 150, 150), 1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(bar, label, ((x0 + x1 - tw) // 2, (y0 + y1 + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return bar

    def render(self):
        # only boxes each frame once, so pausing doesn't feed the same frame
        # to the background subtractors over and over
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
            t = self.pos / self.fps
            txt = (f"{os.path.basename(self.paths[self.i])} "
                   f"[{self.i + 1}/{len(self.paths)}]  frame {self.pos + 1}/{self.n}  "
                   f"{t:.1f}s  x{self.speed:g}  {'PAUSED' if self.paused else 'PLAYING'}")
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
            name = os.path.basename(self.paths[idx])
            if len(name) > 48:
                name = name[:45] + "..."
            cv2.putText(f, name, (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

    def sync_trackbar(self):
        val = int(self.pos / max(self.n - 1, 1) * 1000)
        self.updating_bar = True
        cv2.setTrackbarPos("Seek", WIN, val)
        self.updating_bar = False

    # ---------- main loop ----------
    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        cv2.createTrackbar("Seek", WIN, 0, 1000, self.on_trackbar)

        keymap = {
            ord(" "): "play", ord("a"): "stepb", ord("d"): "stepf", ord("j"): "back10",
            ord("l"): "fwd10", ord("p"): "prev", ord("n"): "next", ord("-"): "slower",
            ord("+"): "faster", ord("="): "faster", ord("i"): "info",
            ord("f"): "flip", ord("o"): "boxes", ord("r"): "loop", ord("v"): "list",
        }

        while not self.quit:
            t0 = time.perf_counter()

            if self.seek_req is not None:
                self.seek(self.seek_req / 1000 * (self.n - 1))
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

            frame_time = 1.0 / (self.fps * self.speed)
            wait = max(1, int((frame_time - (time.perf_counter() - t0)) * 1000))
            if self.paused:
                wait = 30
            key = cv2.waitKey(wait) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in keymap:
                self.do(keymap[key])

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    paths = collect(sys.argv[1:] or [VIDEO_PATH])
    if not paths:
        sys.exit(f"No videos found in {VIDEO_PATH!r}. Set VIDEO_PATH at the top of the file.")
    Player(paths).run()