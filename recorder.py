"""Recording manager — captures raw XYZ samples to binary alongside live inference.

Two-stage queue: feed() (called from W1) appends to a stage-1 list under a
brief lock, no I/O. Once FLUSH_SIZE samples accumulate, the chunk is pushed
to a writer queue drained by a dedicated thread that flushes ~1×/sec.
Keeps W1 off the disk so read loop stays on the 7812 Hz sensor rate.

Binary format: raw float32 little-endian, 3 channels (x_g, y_g, z_g) interleaved,
no header. `.bin` extension. Sample count = file_size // 12. Append mode works
trivially — just open "ab" and write more bytes. Legacy `.csv` files in data/
are still readable by trainer.py for back-compat; new recordings always go to `.bin`.
"""

import os
import queue
import re
import threading
import time

import numpy as np


DATA_DIR = "data"
SAMPLE_RATE = 7812
MIN_SAMPLES = 2604              # one full inference window — must match WINDOW_SIZE
FLUSH_SIZE = SAMPLE_RATE        # ~1 s of samples per writer-queue chunk
MAX_WRITER_Q = 3                # writer-queue cap; drop-oldest with warn beyond
WRITER_JOIN_TIMEOUT_S = 3.0     # bound how long cancel waits for the writer to drain
SAMPLE_BYTES = 4 * 3            # float32 * 3 channels per sample
NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")
_SENTINEL = object()            # writer-loop "stop draining" marker


def sanitize_name(name: str) -> str:
    return NAME_RE.sub("_", (name or "").strip()).strip("_")[:64]


def _count_csv_samples(path: str) -> int:
    """Legacy: count data rows in a .csv (subtracts the header)."""
    with open(path, "r", newline="") as f:
        rows = sum(1 for _ in f)
    return max(0, rows - 1)


def list_existing_labels(data_dir: str = DATA_DIR):
    """Returns [{name, samples, file_size, mtime, path}] for every data/*.bin
    and any back-compat data/*.csv that doesn't have a .bin twin.

    `.bin` sample count is O(1) (file_size // SAMPLE_BYTES). `.csv` falls back
    to a line scan for legacy files.
    """
    if not os.path.isdir(data_dir):
        return []

    by_name = {}    # name -> entry; .bin overrides .csv of the same name
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".bin"):
            ext = ".bin"
        elif fname.endswith(".csv"):
            ext = ".csv"
        else:
            continue
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            st = os.stat(path)
            name = fname[:-4]
            samples = (st.st_size // SAMPLE_BYTES) if ext == ".bin" else _count_csv_samples(path)
            entry = {
                "name": name,
                "samples": samples,
                "file_size": st.st_size,
                "mtime": st.st_mtime,
                "path": path,
            }
            if ext == ".bin" or name not in by_name:
                by_name[name] = entry
        except OSError:
            continue
    return [by_name[n] for n in sorted(by_name)]


def delete_label(data_dir: str, name: str):
    """Delete the recording files for `name` (both .bin and legacy .csv twin).
    `name` must match what `list_existing_labels` returns — we DO NOT re-sanitize
    it (that would mangle e.g. leading underscores). Instead we reject path
    separators and confirm the resolved path lives directly in data_dir.
    Returns the list of removed filenames. Raises ValueError on an invalid
    name or if nothing exists."""
    name = (name or "").strip()
    if not name or "/" in name or "\\" in name or "\x00" in name or name in (".", ".."):
        raise ValueError("invalid name")
    data_dir_abs = os.path.abspath(data_dir)
    removed = []
    for ext in (".bin", ".csv"):
        path = os.path.abspath(os.path.join(data_dir, f"{name}{ext}"))
        # Guard against traversal: the file must sit directly in data_dir.
        if os.path.dirname(path) != data_dir_abs:
            raise ValueError("invalid name")
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed.append(os.path.basename(path))
            except OSError as e:
                raise ValueError(f"could not delete {os.path.basename(path)}: {e}")
    if not removed:
        raise ValueError(f"no recording named {name!r}")
    return removed


class RecordingSession:
    def __init__(self, name, port, target_samples, mode, file_path):
        self.name = name
        self.port = port
        self.target_samples = int(target_samples)
        self.mode = mode                       # "append" | "overwrite"
        self.file_path = file_path
        self.started_ts = time.time()
        self.samples_written = 0               # advances in feed(); = samples committed to recording
        self.status = "active"                 # "active" | "complete" | "cancelled" | "error"
        self.error = None

    def to_dict(self):
        elapsed = time.time() - self.started_ts
        progress = (self.samples_written / self.target_samples) if self.target_samples else 0.0
        return {
            "name": self.name,
            "port": self.port,
            "target_samples": self.target_samples,
            "samples_written": self.samples_written,
            "mode": self.mode,
            "status": self.status,
            "error": self.error,
            "file_path": self.file_path,
            "started_ts": self.started_ts,
            "elapsed_s": elapsed,
            "progress": min(progress, 1.0),
            "sample_rate": SAMPLE_RATE,
        }


class RecordingManager:
    """Thread-safe singleton driving at most one recording session at a time."""

    def __init__(self, data_dir: str = DATA_DIR):
        self._lock = threading.Lock()
        self._session = None
        self._stage1_chunks = []
        self._stage1_n = 0
        self._writer_queue = None
        self._writer_thread = None
        self._last_finished = None
        self.data_dir = data_dir

    def start(self, name, target_samples, port, mode):
        clean_name = sanitize_name(name)
        if not clean_name:
            raise ValueError("name must contain at least one alphanumeric character")

        target_samples = int(target_samples)
        if target_samples < MIN_SAMPLES:
            raise ValueError(f"target_samples must be >= {MIN_SAMPLES} (one inference window)")

        if mode not in ("append", "overwrite"):
            raise ValueError("mode must be 'append' or 'overwrite'")

        with self._lock:
            if self._session is not None and self._session.status == "active":
                raise RuntimeError("a recording is already in progress")

            os.makedirs(self.data_dir, exist_ok=True)
            file_path = os.path.join(self.data_dir, f"{clean_name}.bin")

            open_mode = "ab" if mode == "append" else "wb"
            f = open(file_path, open_mode)

            session = RecordingSession(clean_name, port, target_samples, mode, file_path)
            self._session = session
            self._stage1_chunks = []
            self._stage1_n = 0
            # Unbounded queue; feed() self-enforces a soft cap via drop-oldest.
            self._writer_queue = queue.Queue()
            self._last_finished = None

            t = threading.Thread(target=self._writer_loop,
                                 args=(f, session, self._writer_queue),
                                 daemon=True,
                                 name=f"recorder-writer-{clean_name}")
            self._writer_thread = t
            t.start()
            return session.to_dict()

    def feed(self, port: str, samples: np.ndarray):
        """Hot path — called from W1 on every Modbus read. NO disk I/O.

        Append the chunk to a stage-1 list, bump samples_written, and only
        push to the writer queue when we've accumulated FLUSH_SIZE samples
        (or when the session has reached its target — a final partial chunk).
        """
        s = self._session
        if s is None or s.status != "active" or s.port != port:
            return

        chunks_to_flush = None
        final = False
        wq = None

        with self._lock:
            s = self._session
            if s is None or s.status != "active" or s.port != port:
                return

            remaining = s.target_samples - s.samples_written
            if remaining <= 0:
                return

            chunk = samples if samples.shape[0] <= remaining else samples[:remaining]
            self._stage1_chunks.append(chunk)
            self._stage1_n += int(chunk.shape[0])
            s.samples_written += int(chunk.shape[0])
            target_reached = s.samples_written >= s.target_samples

            if self._stage1_n >= FLUSH_SIZE or target_reached:
                chunks_to_flush = self._stage1_chunks
                self._stage1_chunks = []
                self._stage1_n = 0
                final = target_reached
                wq = self._writer_queue

        # vstack + queue I/O OUTSIDE the lock so W1 stalls only on the trivial
        # under-lock work.
        if chunks_to_flush is None:
            return

        chunk = chunks_to_flush[0] if len(chunks_to_flush) == 1 else np.vstack(chunks_to_flush)

        # Drop-oldest with warning if writer fell behind. Mirrors the inference
        # window queue's overflow handling in sensor_reader.RawWindowReader and
        # raw_data_logger.py:99-102.
        while wq.qsize() >= MAX_WRITER_Q:
            print("[recorder] Warning! Writer queue overwrite (disk too slow?)")
            try:
                wq.get_nowait()
            except queue.Empty:
                break

        wq.put(chunk)
        if final:
            wq.put(_SENTINEL)

    def cancel(self):
        """Mark the session cancelled and ask the writer to drain & exit.

        Blocks up to WRITER_JOIN_TIMEOUT_S waiting for the writer to flush
        whatever it's already accepted, so by the time this returns the
        last_finished snapshot reports the truthful on-disk sample count.
        """
        with self._lock:
            s = self._session
            if s is None or s.status != "active":
                return self._last_finished
            s.status = "cancelled"
            chunks = self._stage1_chunks
            self._stage1_chunks = []
            self._stage1_n = 0
            wq = self._writer_queue
            wt = self._writer_thread

        # Hand the leftover stage-1 buffer to the writer, then nudge it to exit.
        if chunks:
            chunk = chunks[0] if len(chunks) == 1 else np.vstack(chunks)
            wq.put(chunk)
        wq.put(_SENTINEL)

        if wt is not None:
            wt.join(timeout=WRITER_JOIN_TIMEOUT_S)
        return self._last_finished

    def status(self):
        with self._lock:
            if self._session is not None:
                return self._session.to_dict()
            return self._last_finished

    def _writer_loop(self, f, session, wq):
        """Stage-2 thread: drains the queue, writes binary, flushes once per chunk.

        `ndarray.tofile` is a C-level write that releases the GIL during the
        actual bytes-to-disk handoff — that's the whole point of the binary
        format. Pure-Python CSV formatting (the previous approach) held the
        GIL for the entire chunk and starved the reader's modbus poll loop.
        """
        try:
            while True:
                try:
                    chunk = wq.get(timeout=0.1)
                except queue.Empty:
                    continue
                if chunk is _SENTINEL:
                    break
                chunk.astype(np.float32, copy=False).tofile(f)
                f.flush()
        except Exception as e:
            with self._lock:
                if session.error is None:
                    session.error = str(e)
        finally:
            try:
                f.flush()
                f.close()
            except Exception:
                pass

            with self._lock:
                if session.status == "active":
                    if session.samples_written >= session.target_samples:
                        session.status = "complete"
                    else:
                        session.status = "error"
                        if session.error is None:
                            session.error = "writer exited before reaching target"
                self._last_finished = session.to_dict()
                if self._session is session:
                    self._session = None
                    self._writer_queue = None
                    self._writer_thread = None
                    self._stage1_chunks = []
                    self._stage1_n = 0
