import threading
import time

import numpy as np


WINDOW_SIZE      = 3906
SAMPLE_RATE      = 7812
DISPLAY_POINTS   = 256
MIN_RAW_SAMPLES  = 32       # floor for the user-set raw window (savgol needs a few)
RING_CAPACITY    = SAMPLE_RATE * 2     # ~2 s of history per port
FFT_INTERVAL_S   = 1.0 / 4.0           # recompute FFT ~6 Hz (the old window rate)

SAVGOL_WINDOW    = 11
SAVGOL_POLYORDER = 3


def _savgol_coeffs(window_length, polyorder):
    # Savitzky-Golay smoothing coeffs (centre row), zero scipy deps. Derivation
    # in docs/modules.md; equivalent to scipy.signal.savgol_coeffs(W, P).
    if window_length % 2 == 0 or window_length <= polyorder:
        raise ValueError("window_length must be odd and > polyorder")
    half = window_length // 2
    i = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(i, polyorder + 1, increasing=True)
    coeffs = np.linalg.pinv(A)[0]
    return coeffs.astype(np.float32)


SAVGOL_COEFFS = _savgol_coeffs(SAVGOL_WINDOW, SAVGOL_POLYORDER)
SAVGOL_HALF   = SAVGOL_WINDOW // 2


def _savgol(axis):
    # Edge-replicated padding so ends aren't pulled toward zero by convolve's
    # implicit zero-padding. Same-length output.
    padded = np.pad(axis, SAVGOL_HALF, mode='edge')
    return np.convolve(padded, SAVGOL_COEFFS, mode='valid')


def _load_fft_backend():
    # Returns (name, make_plan, run_plan). make_plan(N) → opaque handle passed
    # to run_plan(handle, axis) per axis. pyfftw with cached plans (NEON on
    # aarch64); numpy.fft fallback.
    try:
        import pyfftw

        def make_plan(n):
            in_buf  = pyfftw.empty_aligned(n, dtype='float32')
            out_buf = pyfftw.empty_aligned(n // 2 + 1, dtype='complex64')
            plan = pyfftw.FFTW(in_buf, out_buf,
                               direction='FFTW_FORWARD',
                               flags=('FFTW_MEASURE',),
                               threads=1)
            return (in_buf, out_buf, plan)

        def run_plan(handle, axis):
            in_buf, out_buf, plan = handle
            in_buf[:] = axis
            plan()
            return np.abs(out_buf)

        return "pyfftw", make_plan, run_plan
    except ImportError:
        def make_plan(n):
            return n

        def run_plan(handle, axis):
            return np.abs(np.fft.rfft(axis)).astype(np.float32)

        return "numpy", make_plan, run_plan


FFT_BACKEND_NAME, _make_fft_plan, _run_fft_plan = _load_fft_backend()
print(f"[waveform] FFT backend = {FFT_BACKEND_NAME}")


class WaveformBus:
    # Kept separate from state.SnapshotBus so prediction/waveform SSE don't
    # false-wake each other.

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


class WaveformAggregator:
    # Per-port ring buffer + latest snapshot. append() from the raw drain
    # thread, render_tick() from the display tick, snapshot() from Flask.

    def __init__(self, ports, window_size=WINDOW_SIZE, sample_rate=SAMPLE_RATE,
                 display_points=DISPLAY_POINTS, ring_capacity=RING_CAPACITY):
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.display_points = display_points
        self._ports = list(ports)
        self._ring_cap = ring_capacity

        # How many recent samples raw mode shows (user-tunable). Ring holds ~2 s.
        self.raw_max_samples = sample_rate                 # up to 1 s
        self.raw_samples = min(window_size // 2, self.raw_max_samples)

        # FFT bin count (user-tunable). bin = sample_rate/window_size ≈ 3 Hz.
        self.bin_hz = sample_rate / window_size
        self.fft_max_bins = window_size // 2
        self.fft_bins = min(display_points // 2, self.fft_max_bins)

        self._lock = threading.Lock()
        self._latest = {p: None for p in ports}
        self._raw_seq = {p: 0 for p in ports}
        self._fft_seq = {p: 0 for p in ports}

        # _ring_w = next write index, _ring_n = total ever written (monotonic),
        # _rendered_n = _ring_n at last render (skip recompute when unchanged).
        self._ring = {p: np.zeros((ring_capacity, 3), dtype=np.float32) for p in ports}
        self._ring_w = {p: 0 for p in ports}
        self._ring_n = {p: 0 for p in ports}
        self._rendered_n = {p: 0 for p in ports}
        self._last_fft_t = {p: 0.0 for p in ports}

        self._plans = {p: _make_fft_plan(window_size) for p in ports}

        self.raw_axis = self._make_raw_axis(self.raw_samples)
        self.freq_axis_hz = self._make_freq_axis(self.fft_bins)

    def _make_freq_axis(self, bins):
        return (np.arange(1, bins + 1, dtype=np.float32) * self.bin_hz).tolist()

    def _make_raw_axis(self, n):
        # 'Samples ago' axis, 0 = latest; client reverses so it sits on the right.
        m = min(n, self.display_points)
        return np.linspace(0.0, n - 1, m, dtype=np.float32).tolist()

    def set_fft_max_hz(self, hz):
        bins = int(round(float(hz) / self.bin_hz))
        bins = max(1, min(bins, self.fft_max_bins))
        axis = self._make_freq_axis(bins)
        with self._lock:
            self.fft_bins = bins
            self.freq_axis_hz = axis
        return bins * self.bin_hz

    def set_raw_samples(self, n):
        n = max(MIN_RAW_SAMPLES, min(int(n), self.raw_max_samples))
        axis = self._make_raw_axis(n)
        with self._lock:
            self.raw_samples = n
            self.raw_axis = axis
        return n

    def append(self, port, chunk):
        # Write a fresh (k, 3) chunk into the ring. Cheap memcpy.
        if port not in self._ring:
            return
        k = chunk.shape[0]
        cap = self._ring_cap
        with self._lock:
            ring = self._ring[port]
            w = self._ring_w[port]
            if k >= cap:
                ring[:] = chunk[-cap:]
                self._ring_w[port] = 0
            elif w + k <= cap:
                ring[w:w + k] = chunk
                self._ring_w[port] = (w + k) % cap
            else:
                first = cap - w
                ring[w:] = chunk[:first]
                ring[:k - first] = chunk[first:]
                self._ring_w[port] = k - first
            self._ring_n[port] += k

    def _tail_locked(self, port, n):
        # Last `n` samples (oldest→latest), honoring wrap. None if < n written.
        total = self._ring_n[port]
        if total < n:
            return None
        cap = self._ring_cap
        ring = self._ring[port]
        start = (self._ring_w[port] - n) % cap
        if start + n <= cap:
            return ring[start:start + n].copy()
        return np.concatenate([ring[start:], ring[:(start + n) - cap]], axis=0)

    def render_tick(self):
        # Recompute per-port snapshots. Raw every tick (scroll); FFT ~every
        # FFT_INTERVAL_S. Skips ports with no new samples. Returns True if
        # anything changed (caller bumps the bus).
        now = time.monotonic()
        with self._lock:
            raw_n = self.raw_samples
            fft_bins = self.fft_bins
            jobs = {}
            for port in self._ports:
                if self._ring_n[port] == self._rendered_n[port]:
                    continue
                raw_tail = self._tail_locked(port, raw_n)
                if raw_tail is None:
                    continue            # ring not full enough yet → 'waiting…'
                do_fft = (now - self._last_fft_t[port]) >= FFT_INTERVAL_S
                fft_tail = self._tail_locked(port, self.window_size) if do_fft else None
                if fft_tail is not None:
                    self._last_fft_t[port] = now
                self._rendered_n[port] = self._ring_n[port]
                jobs[port] = (raw_tail, fft_tail)
        if not jobs:
            return False

        # Heavy compute OUTSIDE the lock.
        results = {}
        for port, (raw_tail, fft_tail) in jobs.items():
            raw_xyz = self._raw_from_tail(raw_tail)
            fft_xyz = None
            if fft_tail is not None:
                try:
                    fft_xyz = self._fft_from_tail(self._plans[port], fft_tail, fft_bins)
                except Exception as e:
                    print(f"[waveform] fft compute failed for {port}: {e}")
            results[port] = (raw_xyz, fft_xyz)

        with self._lock:
            for port, (raw_xyz, fft_xyz) in results.items():
                self._raw_seq[port] += 1
                prev = self._latest.get(port) or {}
                entry = {
                    "raw_seq": self._raw_seq[port],
                    "raw": raw_xyz,
                    "ts": time.time(),
                    "fft_seq": prev.get("fft_seq", 0),
                    "fft": prev.get("fft"),
                }
                if fft_xyz is not None:
                    self._fft_seq[port] += 1
                    entry["fft_seq"] = self._fft_seq[port]
                    entry["fft"] = fft_xyz
                self._latest[port] = entry
        return True

    def _raw_from_tail(self, tail):
        # Smoothed, decimated, reversed ('samples ago') view → display_points.
        n = tail.shape[0]
        m = min(n, self.display_points)
        x_ago = np.linspace(0.0, n - 1, m)
        src = np.clip((n - 1 - x_ago).round().astype(int), 0, n - 1)
        out = {}
        for i, name in enumerate(("x", "y", "z")):
            axis = np.ascontiguousarray(tail[:, i], dtype=np.float32)
            sm = _savgol(axis)
            out[name] = sm[src].astype(np.float32).tolist()
        return out

    def _fft_from_tail(self, plan, tail, bins):
        # Magnitude of the first `bins` rFFT bins (DC excluded), per axis.
        out = {}
        for i, name in enumerate(("x", "y", "z")):
            axis = np.ascontiguousarray(tail[:, i], dtype=np.float32)
            mag = _run_fft_plan(plan, axis)
            out[name] = mag[1:1 + bins].astype(np.float32).tolist()
        return out

    def snapshot(self):
        with self._lock:
            ports_out = {}
            for port in self._ports:
                entry = self._latest.get(port)
                ports_out[port] = entry if entry is not None else {
                    "raw_seq": 0, "fft_seq": 0, "ts": None, "raw": None, "fft": None,
                }
            freq_axis = self.freq_axis_hz
            fft_bins = self.fft_bins
            raw_axis = self.raw_axis
            raw_n = self.raw_samples
        return {
            "ports": ports_out,
            "raw_axis": raw_axis,
            "freq_axis_hz": freq_axis,
            "fft_backend": FFT_BACKEND_NAME,
            "sample_rate": self.sample_rate,
            "window_size": self.window_size,
            "raw_samples": raw_n,
            "raw_max_samples": self.raw_max_samples,
            "fft_bins": fft_bins,
            "fft_bin_hz": self.bin_hz,
            "fft_max_bins": self.fft_max_bins,
            "now": time.time(),
        }
