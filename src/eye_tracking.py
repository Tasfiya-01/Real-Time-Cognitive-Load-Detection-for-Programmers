"""Webcam blink-rate tracker for the dashboard's Bug Vulnerability Index (eye score).

Detector (chosen automatically, best first):
  1. MediaPipe Face Landmarker (478 face-mesh landmarks) + eye aspect ratio (EAR)
       EAR = (|p2 - p6| + |p3 - p5|) / (2 |p1 - p4|); eye counted as closed when EAR falls below
       75% of the person's own open-eye EAR (90th percentile of the last 5 s), because open-eye EAR
       differs between people (about 0.22-0.35). --ear 0.25 sets a fixed threshold instead.
  2. OpenCV Haar cascades (fallback if MediaPipe is not installed): eyes not found while the face is found
A blink = eye closed for 50-500 ms (longer closures are not counted as blinks).
Blink rate = blinks in the last 60 s (scaled up during the first minute), sent to the backend every 5 s.

Usage:  python src\\eye_tracking.py                 (window with overlay, press Q to quit)
        python src\\eye_tracking.py --detector opencv
        python src\\eye_tracking.py --camera 1 --no-window
Needs:  pip install opencv-python requests   (optional, better: pip install mediapipe)
The MediaPipe model file (about 4 MB) is downloaded once to models/face_landmarker.task.
"""
import argparse, threading, time, urllib.request
from collections import deque
from pathlib import Path
import numpy as np, cv2, requests

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = Path("models/face_landmarker.task")
# Face-mesh landmark indices p1..p6 for each eye (outer corner, two upper lid, inner corner, two lower lid)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def ear(pts):
    """Eye aspect ratio from six (x, y) points p1..p6."""
    p1, p2, p3, p4, p5, p6 = [np.asarray(p, float) for p in pts]
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4) + 1e-9)


class BlinkCounter:
    """Turns a stream of (time, eye_closed) samples into blinks and a blink rate."""
    def __init__(self, min_s=0.05, max_s=0.5, window_s=60.0):
        self.min_s, self.max_s, self.window_s = min_s, max_s, window_s
        self.closed_since = None
        self.blinks = deque()
        self.start = None
        self.total = 0

    def update(self, t, closed):
        if self.start is None:
            self.start = t
        if closed is None:                      # no face: reset the closure, count nothing
            self.closed_since = None
            return
        if closed and self.closed_since is None:
            self.closed_since = t
        elif not closed and self.closed_since is not None:
            dur = t - self.closed_since
            self.closed_since = None
            if self.min_s <= dur <= self.max_s:
                self.blinks.append(t); self.total += 1
        while self.blinks and t - self.blinks[0] > self.window_s:
            self.blinks.popleft()

    def rate(self, t):
        """Blinks per minute over the last 60 s; None during the first 10 s."""
        if self.start is None or t - self.start < 10:
            return None
        span = min(self.window_s, t - self.start)
        return round(len(self.blinks) * 60.0 / span, 1)


def eye_load(rate):
    # same hand-set bands as the dashboard eye score (design concept, not validated)
    if rate is None: return None
    if rate > 25 or rate < 8: return "High"
    if rate > 18 or rate < 12: return "Medium"
    return "Low"


class AdaptiveThreshold:
    """Per-person blink threshold: ratio x open-eye EAR (90th percentile of recent frames)."""
    def __init__(self, ratio=0.75, window_s=5.0):
        self.ratio, self.window_s, self.hist = ratio, window_s, deque()
        self.baseline = None
    def __call__(self, t, e):
        self.hist.append((t, e))
        while t - self.hist[0][0] > self.window_s: self.hist.popleft()
        if len(self.hist) >= 10:
            self.baseline = float(np.percentile([v for _, v in self.hist], 90))
        return None if self.baseline is None else self.ratio * self.baseline


class MediaPipeDetector:
    name = "MediaPipe Face Mesh + EAR"
    def __init__(self, ear_threshold):
        import mediapipe as mp
        self.mp, self.fixed, self.legacy = mp, ear_threshold, None
        self.adapt, self.thr, self.ears = AdaptiveThreshold(), None, []
        if hasattr(mp, "solutions"):            # older MediaPipe (< 0.10.30): Face Mesh solution
            self.legacy = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)
            return
        from mediapipe.tasks.python import BaseOptions, vision
        if not MODEL_PATH.exists():
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading MediaPipe face model (once)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                                            running_mode=vision.RunningMode.VIDEO, num_faces=1)
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self.t0 = time.time()

    def __call__(self, frame):
        """Returns (eye_closed or None if no face, EAR or None)."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.legacy is not None:
            res = self.legacy.process(rgb)
            if not res.multi_face_landmarks: return None, None
            lms = res.multi_face_landmarks[0].landmark
        else:
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.lm.detect_for_video(img, int((time.time() - self.t0) * 1000))
            if not res.face_landmarks: return None, None
            lms = res.face_landmarks[0]
        P = lambda idx: [(lms[i].x * w, lms[i].y * h) for i in idx]
        e = (ear(P(LEFT_EYE)) + ear(P(RIGHT_EYE))) / 2
        self.ears.append(e)
        self.thr = self.fixed if self.fixed else self.adapt(time.time(), e)
        if self.thr is None: return False, e          # first frames: learn the open-eye EAR
        return e < self.thr, e


class HaarDetector:
    name = "OpenCV Haar cascades"
    def __init__(self, *_):
        self.face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        if self.face.empty() or self.eye.empty():
            raise RuntimeError("OpenCV Haar cascade files not found. Install MediaPipe (pip install mediapipe) "
                               "or opencv-python 4.x (pip install \"opencv-python<5\").")

    def __call__(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0: return None, None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        top = gray[y:y + h // 2 + h // 10, x:x + w]          # eyes are in the upper half of the face
        eyes = self.eye.detectMultiScale(top, 1.1, 6, minSize=(w // 10, w // 10))
        return len(eyes) == 0, None


def make_detector(kind, thr):
    if kind in ("auto", "mediapipe"):
        try:
            return MediaPipeDetector(thr)
        except Exception as e:
            if kind == "mediapipe": raise
            print(f"MediaPipe not available ({type(e).__name__}: {e}); using OpenCV Haar cascades.")
    return HaarDetector()


def post(url, payload):
    try: requests.post(url, json=payload, timeout=1)
    except Exception: pass


def run(args):
    det = make_detector(args.detector, args.ear)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Webcam not found. Try --camera 1."); return
    bc, last_post = BlinkCounter(), 0.0
    print(f"Eye tracking started ({det.name}). Sending to {args.backend}/update_eye every 5 s. Press Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok: break
        t = time.time()
        closed, e = det(frame)
        bc.update(t, closed)
        rate = bc.rate(t)
        if t - last_post >= 5 and rate is not None:
            last_post = t
            threading.Thread(target=post, daemon=True, args=(args.backend + "/update_eye",
                             {"blink_rate": rate, "eye_load": eye_load(rate), "detector": det.name})).start()
        if not args.no_window:
            col = {"High": (0, 0, 255), "Medium": (0, 165, 255), "Low": (0, 200, 0)}.get(eye_load(rate), (200, 200, 200))
            lines = [f"Blink rate: {rate if rate is not None else 'measuring...'} /min",
                     f"Eye load:   {eye_load(rate) or '-'}",
                     f"Blinks: {bc.total}   Face: {'yes' if closed is not None else 'no'}"
                     + (f"   EAR: {e:.2f}" if e is not None else "")
                     + (f"  thr: {det.thr:.2f}" if getattr(det, "thr", None) else ""),
                     f"{det.name} | Q to quit"]
            for k, s in enumerate(lines):
                cv2.putText(frame, s, (15, 30 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.65 if k < 2 else 0.5,
                            col if k < 2 else (220, 220, 220), 2 if k < 2 else 1)
            cv2.imshow("Eye tracking - blink rate", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release(); cv2.destroyAllWindows()
    print(f"Session: {bc.total} blinks, last rate {bc.rate(time.time())}/min")
    if getattr(det, "ears", None):
        a = np.array(det.ears)
        print(f"EAR open-eye (90th pct) {np.percentile(a, 90):.3f}, median {np.median(a):.3f}, "
              f"10th pct {np.percentile(a, 10):.3f}; threshold {'fixed ' + str(det.fixed) if det.fixed else 'adaptive, last %.3f' % (det.thr or 0)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--detector", choices=["auto", "mediapipe", "opencv"], default="auto")
    ap.add_argument("--ear", type=float, default=None, help="fixed EAR blink threshold (default: adaptive)")
    ap.add_argument("--backend", default="http://localhost:5000")
    ap.add_argument("--no-window", action="store_true")
    run(ap.parse_args())