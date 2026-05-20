"""Trainer — computes class prototypes from recorded data (.bin or legacy .csv).

"Training" here is just: for each selected label, run every (WINDOW, 3) window
through the backbone, average the resulting embeddings to a single 128-d
prototype, write all prototypes to `classifier_head.json`, and ask the live
`InferenceWorker` to hot-reload its head. No TF, no gradients, takes seconds.

Singleton + background thread + status pattern mirrors `recorder.py`.
"""

import os
import random
import threading
import time

import numpy as np
import pandas as pd

from classifier import ClassifierHead, DEFAULT_HEAD_PATH, PALETTE_SIZE


WINDOW_SIZE = 2604               # must match train_backbone.py / sensor_reader.py
MIN_WINDOWS = 6                  # 6 × 2604 = 2.0 s of recording — permissive floor
# Stride between training windows. WINDOW_SIZE // 4 → 75 % overlap → 4 distinct
# phase alignments per gesture cycle (offsets 0, 651, 1302, 1953 samples mod
# WINDOW_SIZE). Yields ~4× more training samples per recording than the old
# non-overlapping slicing. 50 % overlap was tempting (only 2× compute) but
# 2604 / 1302 = 2, so 50 % hop hits only 2 phase positions — strictly worse
# phase coverage than 25 % overlap. 75 % is the sweet spot for this geometry.
TRAIN_HOP = WINDOW_SIZE // 4


SAMPLE_BYTES = 4 * 3            # float32 * 3 channels — must match recorder.SAMPLE_BYTES


def _label_path(data_dir: str, label: str):
    """Return (path, ext) for whichever format exists. .bin wins if both do."""
    bin_path = os.path.join(data_dir, f"{label}.bin")
    if os.path.exists(bin_path):
        return bin_path, ".bin"
    csv_path = os.path.join(data_dir, f"{label}.csv")
    if os.path.exists(csv_path):
        return csv_path, ".csv"
    return None, None


def load_label_windows(data_dir: str, label: str) -> np.ndarray:
    """Read data/<label>.bin or .csv → (N_windows, WINDOW_SIZE, 3) float32.

    Windows are extracted with TRAIN_HOP stride (75 % overlap), giving ~4×
    the training samples a non-overlapping slice would produce. The benefit
    is twofold: lower variance on the prototype mean estimate, and broader
    phase coverage so the prototype matches inference-time windows
    regardless of where the live gesture happens to start within the window.

    Returns an empty (0, WINDOW_SIZE, 3) array if the file is missing or
    has fewer than WINDOW_SIZE samples.
    """
    path, ext = _label_path(data_dir, label)
    if path is None:
        return np.empty((0, WINDOW_SIZE, 3), dtype=np.float32)

    if ext == ".bin":
        arr = np.fromfile(path, dtype=np.float32).reshape(-1, 3)
    else:
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df = df.drop(columns=["timestamp"])
        if df.shape[1] != 3:
            raise ValueError(f"{path}: expected 3 columns after dropping timestamp, got {df.shape[1]}")
        arr = df.values.astype(np.float32)

    if arr.shape[0] < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE, 3), dtype=np.float32)

    # Sliding-window extraction. np.stack copies the slices so we don't ship
    # strided views into downstream code; the embed() pass would have to
    # copy anyway since TFLite needs contiguous buffers.
    last_start = arr.shape[0] - WINDOW_SIZE
    starts = range(0, last_start + 1, TRAIN_HOP)
    return np.stack([arr[s:s + WINDOW_SIZE] for s in starts])


def count_windows(data_dir: str, label: str) -> int:
    """Cheap window count. O(1) for .bin via file_size; line scan for legacy .csv."""
    path, ext = _label_path(data_dir, label)
    if path is None:
        return 0
    if ext == ".bin":
        samples = os.path.getsize(path) // SAMPLE_BYTES
    else:
        with open(path, "r", newline="") as f:
            samples = max(0, sum(1 for _ in f) - 1)
    return samples // WINDOW_SIZE


class TrainingSession:
    def __init__(self, selected_labels):
        self.selected_labels = list(selected_labels)
        self.labels_total = len(self.selected_labels)
        self.labels_done = 0
        self.current_label = None
        self.status = "active"         # "active" | "complete" | "cancelled" | "error"
        self.error = None
        self.started_ts = time.time()
        self.head_path = None

    def to_dict(self):
        elapsed = time.time() - self.started_ts
        progress = (self.labels_done / self.labels_total) if self.labels_total else 0.0
        return {
            "selected_labels": list(self.selected_labels),
            "labels_total": self.labels_total,
            "labels_done": self.labels_done,
            "current_label": self.current_label,
            "status": self.status,
            "error": self.error,
            "started_ts": self.started_ts,
            "elapsed_s": elapsed,
            "progress": min(progress, 1.0),
            "head_path": self.head_path,
        }


class TrainerManager:
    """At most one training session at a time."""

    def __init__(self, data_dir: str, head_path: str = DEFAULT_HEAD_PATH):
        self._lock = threading.Lock()
        self._session = None
        self._thread = None
        self._cancel_event = None
        self._last_finished = None
        self.data_dir = data_dir
        self.head_path = head_path

    def start(self, inference_worker, selected_labels):
        labels = [str(l).strip() for l in (selected_labels or []) if str(l).strip()]
        if len(labels) < 2:
            raise ValueError("select at least 2 labels")

        # Resolve files + window counts before spawning the thread so we can
        # surface friendly errors synchronously to the HTTP caller.
        for lbl in labels:
            path, _ext = _label_path(self.data_dir, lbl)
            if path is None:
                raise ValueError(f"label '{lbl}' has no .bin or .csv in {self.data_dir}")
            if count_windows(self.data_dir, lbl) < MIN_WINDOWS:
                raise ValueError(
                    f"label '{lbl}' has < {MIN_WINDOWS} windows — record more via /record"
                )

        if inference_worker is None or getattr(inference_worker, "_interp", None) is None:
            raise RuntimeError("backbone interpreter not loaded — can't train")

        with self._lock:
            if self._session is not None and self._session.status == "active":
                raise RuntimeError("a training session is already in progress")
            session = TrainingSession(labels)
            self._session = session
            self._cancel_event = threading.Event()
            self._last_finished = None
            t = threading.Thread(target=self._run, args=(inference_worker, session, self._cancel_event),
                                 daemon=True, name="trainer")
            self._thread = t
            t.start()
            return session.to_dict()

    def status(self):
        with self._lock:
            if self._session is not None:
                return self._session.to_dict()
            return self._last_finished

    def cancel(self):
        with self._lock:
            if self._session is None or self._session.status != "active":
                return self._last_finished
            self._cancel_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return self._last_finished

    def _run(self, inference_worker, session, cancel_event):
        prototypes = []
        try:
            for label in session.selected_labels:
                if cancel_event.is_set():
                    self._finalize(session, "cancelled")
                    return

                with self._lock:
                    session.current_label = label

                windows = load_label_windows(self.data_dir, label)
                if windows.shape[0] < MIN_WINDOWS:
                    raise ValueError(
                        f"label '{label}' yielded {windows.shape[0]} windows < {MIN_WINDOWS}"
                    )

                feats = np.empty((windows.shape[0], 128), dtype=np.float32)
                for i, w in enumerate(windows):
                    if cancel_event.is_set():
                        self._finalize(session, "cancelled")
                        return
                    feats[i] = inference_worker.embed(w)

                proto = feats.mean(axis=0)
                prototypes.append(proto)

                with self._lock:
                    session.labels_done += 1
                print(f"[trainer] {label}: {windows.shape[0]} windows → prototype "
                      f"(norm={float(np.linalg.norm(proto)):.3f})")

            # Random colour assignment per train. ≤ PALETTE_SIZE labels get
            # distinct slots; beyond that we wrap. Uses Python's RNG, not
            # numpy's, so external numpy seeding can't make the shuffle stable.
            n = len(session.selected_labels)
            if n <= PALETTE_SIZE:
                colors = random.sample(range(PALETTE_SIZE), n)
            else:
                base = list(range(PALETTE_SIZE))
                random.shuffle(base)
                colors = [base[i % PALETTE_SIZE] for i in range(n)]

            head = ClassifierHead(session.selected_labels, np.stack(prototypes),
                                  colors=colors)
            head.save(self.head_path)
            with self._lock:
                session.head_path = self.head_path

            inference_worker.reload_head()
            self._finalize(session, "complete")

        except Exception as e:
            print(f"[trainer] failed: {e}")
            with self._lock:
                session.error = str(e)
            self._finalize(session, "error")

    def _finalize(self, session, status: str):
        with self._lock:
            session.status = status
            session.current_label = None
            self._last_finished = session.to_dict()
            if self._session is session:
                self._session = None
                self._thread = None
                self._cancel_event = None
