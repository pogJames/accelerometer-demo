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
    with open(path, "r", newline="") as f:
        rows = sum(1 for _ in f)
    return max(0, rows - 1)     # minus header


def list_existing_labels(data_dir: str = DATA_DIR):
    # [{name, samples, file_size, mtime, path}] for every data/*.bin and any
    # *.csv without a .bin twin. .bin sample count is O(1); .csv line-scans.
    if not os.path.isdir(data_dir):
        return []

    by_name = {}    # .bin overrides .csv of the same name
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
    # `name` must match list_existing_labels output — we do NOT re-sanitize
    # (that would mangle leading underscores). Reject separators + confirm the
    # resolved path sits directly in data_dir. Returns removed filenames.
    name = (name or "").strip()
    if not name or "/" in name or "\\" in name or "\x00" in name or name in (".", ".."):
        raise ValueError("invalid name")
    data_dir_abs = os.path.abspath(data_dir)
    removed = []
    for ext in (".bin", ".csv"):
        path = os.path.abspath(os.path.join(data_dir, f"{name}{ext}"))
        if os.path.dirname(path) != data_dir_abs:   # traversal guard
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
        self.samples_written = 0
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
    # At most one recording session at a time. See pages/notes.md for the two-stage
    # queue and binary format.

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
        # Hot path — called from W1 on every read. NO disk I/O: append to a
        # stage-1 list, only push to the writer queue at FLUSH_SIZE or target.
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

        while wq.qsize() >= MAX_WRITER_Q:       # drop-oldest if writer fell behind
            print("[recorder] Warning! Writer queue overwrite (disk too slow?)")
            try:
                wq.get_nowait()
            except queue.Empty:
                break

        wq.put(chunk)
        if final:
            wq.put(_SENTINEL)

    def cancel(self):
        # Mark cancelled, hand the writer the leftover buffer, then block up to
        # WRITER_JOIN_TIMEOUT_S so last_finished reports the true on-disk count.
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
        # Stage-2: drain queue, write binary, flush per chunk. tofile releases
        # the GIL during the disk handoff — the point of the binary format.
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
