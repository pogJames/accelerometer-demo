import threading
import time
from collections import Counter, deque

from classifier import ClassifierHead, DEFAULT_HEAD_PATH


def load_class_labels():
    # Startup hint only; authoritative label per prediction comes from the
    # head live at invoke time. See docs/modules.md.
    head = ClassifierHead.load(DEFAULT_HEAD_PATH)
    return head.labels if head else ["untrained"]


CLASS_LABELS = load_class_labels()

DISPLAY_REFRESH_EVERY = 2
ROLLING_WINDOW = 7


class LatestSlot:
    def __init__(self):
        self._lock = threading.Lock()
        self._value = None
        self._ts = 0.0

    def set(self, value):
        with self._lock:
            self._value = value
            self._ts = time.time()

    def get(self):
        with self._lock:
            return self._value, self._ts


class SnapshotBus:
    def __init__(self):
        self._cv = threading.Condition()
        self._seq = 0

    def bump(self):
        with self._cv:
            self._seq += 1
            self._cv.notify_all()

    def current_seq(self):
        with self._cv:
            return self._seq

    def wait_for_change(self, last_seen_seq, timeout=None):
        with self._cv:
            self._cv.wait_for(lambda: self._seq > last_seen_seq, timeout=timeout)
            return self._seq


class RollingPredictions:
    def __init__(self, maxlen=ROLLING_WINDOW, refresh_every=DISPLAY_REFRESH_EVERY,
                 on_latch=None):
        self._lock = threading.Lock()
        self._buf = deque(maxlen=maxlen)
        self._refresh_every = refresh_every
        self._pending = 0
        self._displayable = None
        self._display_seq = 0
        self._on_latch = on_latch

    def push(self, prediction: dict):
        latched = False
        with self._lock:
            self._buf.append(prediction)
            self._pending += 1
            if self._pending >= self._refresh_every:
                self._displayable = self._build_snapshot_locked()
                self._pending = 0
                latched = True
        # Fire OUTSIDE the lock — bus.bump() takes its own lock, don't nest.
        if latched and self._on_latch is not None:
            self._on_latch()

    def _build_snapshot_locked(self):
        if not self._buf:
            return None

        # Majority class, tie-broken by recency (walk deque in reverse).
        counts = Counter(p["class_id"] for p in self._buf)
        top_count = counts.most_common(1)[0][1]
        tied = {cid for cid, c in counts.items() if c == top_count}

        majority_class_id = None
        majority_class_name = None
        for p in reversed(self._buf):
            if p["class_id"] in tied:
                majority_class_id = p["class_id"]
                # Stored class_name reflects the head live when classified, so
                # a mid-session retrain surfaces new labels immediately.
                majority_class_name = p.get("class_name")
                break

        recent = list(self._buf)
        latest = recent[-1]

        self._display_seq += 1
        return {
            "display_seq": self._display_seq,
            "majority_class_id": majority_class_id,
            "majority_class_name": majority_class_name
                if majority_class_name is not None
                else f"class_{majority_class_id}",
            "majority_count": counts[majority_class_id],
            "window_count": len(recent),
            "recent": recent,
            "latest_ts": latest["ts"],
        }

    def displayable(self):
        with self._lock:
            return self._displayable

    def clear(self):
        # Called by app.py when switching active sensor so stale predictions
        # don't influence the new majority; notify SSE to re-render 'waiting…'.
        with self._lock:
            self._buf.clear()
            self._displayable = None
            self._pending = 0
        if self._on_latch is not None:
            self._on_latch()
