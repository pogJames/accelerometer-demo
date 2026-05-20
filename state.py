"""Shared state primitives used by all three workers (project4 — backbone + head)."""

import threading
import time
from collections import Counter, deque

from classifier import ClassifierHead, DEFAULT_HEAD_PATH


def load_class_labels():
    """Resolve class labels from the trained head on disk, or fall back to a
    single 'untrained' placeholder if no head exists yet.

    Only used as a startup hint for templates and the dashboard's colour map.
    The authoritative class name per prediction comes from `InferenceWorker`
    at invoke time, which reads `self.head.labels` directly — so this constant
    being a snapshot of an earlier head is fine.
    """
    head = ClassifierHead.load(DEFAULT_HEAD_PATH)
    return head.labels if head else ["untrained"]


CLASS_LABELS = load_class_labels()

DISPLAY_REFRESH_EVERY = 2
ROLLING_WINDOW = 7


class LatestSlot:
    """Atomic last-value slot. Producers call set(); consumers call get()."""

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
    """Broadcaster that wakes any number of SSE clients whenever a port
    latches a new snapshot.

    Each latch increments a monotonic counter and notifies all waiters.
    SSE generators track their last-seen counter so they don't miss
    notifications even if multiple clients are connected and one of them
    was momentarily slow.
    """

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
        """Block until self._seq > last_seen_seq, or timeout. Returns the
        current sequence number (which may equal last_seen_seq if timed out)."""
        with self._cv:
            self._cv.wait_for(lambda: self._seq > last_seen_seq, timeout=timeout)
            return self._seq


class RollingPredictions:
    """Per-port rolling buffer of the last N inference results plus an
    event-driven 'displayable snapshot' that only refreshes once every
    DISPLAY_REFRESH_EVERY pushes.

    On each latch, calls the optional `on_latch` callback so an external
    SnapshotBus can broadcast to SSE clients.
    """

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
        # Fire the callback OUTSIDE the lock — bus.bump() acquires its own
        # condition's lock, and we don't want to nest.
        if latched and self._on_latch is not None:
            self._on_latch()

    def _build_snapshot_locked(self):
        if not self._buf:
            return None

        # Majority class over the rolling window. Counter.most_common picks one
        # winner deterministically; we tie-break by recency by walking the deque
        # in reverse and preferring the most-recent class among the tied top.
        counts = Counter(p["class_id"] for p in self._buf)
        top_count = counts.most_common(1)[0][1]
        tied = {cid for cid, c in counts.items() if c == top_count}

        majority_class_id = None
        majority_class_name = None
        for p in reversed(self._buf):
            if p["class_id"] in tied:
                majority_class_id = p["class_id"]
                # Use the class_name InferenceWorker stored on the prediction
                # itself — this reflects the head that was live when the
                # window was classified, so a retrain mid-session immediately
                # surfaces the new labels instead of stale CLASS_LABELS.
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
