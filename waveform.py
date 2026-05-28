"""Per-port live-waveform aggregator for the / dashboard.

The waveform worker thread calls `WaveformAggregator.publish(port, window)`
on every window it pulls off the in-process fan-out queue. Both raw and FFT
compute on every call — no throttling. Raw shows the freshest hop (the last
window_size//2 samples), FFT uses the full window so frequency resolution
stays at sample_rate/window_size ≈ 3 Hz/bin.

FFT uses pyfftw with cached FFTW plans (so NEON SIMD on aarch64 actually
runs). Falls back to numpy.fft.rfft if pyfftw isn't installed.
"""

import threading
import time

import numpy as np


WINDOW_SIZE      = 2604
SAMPLE_RATE      = 7812
DISPLAY_POINTS   = 256

SAVGOL_WINDOW    = 11
SAVGOL_POLYORDER = 3


def _savgol_coeffs(window_length, polyorder):
    """Savitzky-Golay smoothing coefficients (middle row), zero scipy deps.

    Standard textbook derivation: build Vandermonde matrix A where A[i, k] =
    i**k for i in [-half, +half], solve the normal equations, return the row
    of (A^T A)^-1 A^T corresponding to the centre point (derivative=0).
    """
    if window_length % 2 == 0 or window_length <= polyorder:
        raise ValueError("window_length must be odd and > polyorder")
    half = window_length // 2
    i = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(i, polyorder + 1, increasing=True)            # (W, P+1)
    # pinv row 0 == coefficients that recover the centre value as a least-
    # squares polynomial fit. Equivalent to scipy.signal.savgol_coeffs(W, P).
    coeffs = np.linalg.pinv(A)[0]
    return coeffs.astype(np.float32)


SAVGOL_COEFFS = _savgol_coeffs(SAVGOL_WINDOW, SAVGOL_POLYORDER)
SAVGOL_HALF   = SAVGOL_WINDOW // 2


def _savgol(axis):
    """Apply the Savitzky-Golay smoother with edge-replicated padding so the
    first/last samples aren't pulled toward zero by np.convolve's implicit
    zero-padding in mode='same'. Returns same-length output."""
    padded = np.pad(axis, SAVGOL_HALF, mode='edge')
    return np.convolve(padded, SAVGOL_COEFFS, mode='valid')


def _load_fft_backend():
    """Return (name, make_plan, run_plan). make_plan(N) returns an opaque
    handle the worker passes to run_plan(handle, axis_array) per axis."""
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


def _decimate_to(arr, n_out):
    """Strided decimation to length n_out. Input len must be >= n_out."""
    stride = arr.shape[0] // n_out
    return arr[:stride * n_out:stride][:n_out]


class WaveformBus:
    """SnapshotBus-equivalent for the waveform stream. Kept separate from
    state.SnapshotBus so prediction and waveform SSE don't false-wake each
    other when only one of them has new data."""

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
    """Per-port latest-window store. publish() is called from the inference
    thread on every window; throttle and compute happen inline. snapshot()
    is called from Flask request threads to read the latest state."""

    def __init__(self, ports, on_publish=None,
                 window_size=WINDOW_SIZE, sample_rate=SAMPLE_RATE,
                 display_points=DISPLAY_POINTS):
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.display_points = display_points
        self._on_publish = on_publish

        # Raw view shows only the freshest hop — the last half of each
        # 2604-sample window, which is exactly the new samples since the
        # previous emit (the other half is the 50% overlap, already shown).
        self.raw_samples = window_size // 2     # 1302 at the default config

        self._lock = threading.Lock()
        self._latest = {p: None for p in ports}
        self._raw_seq = {p: 0 for p in ports}
        self._fft_seq = {p: 0 for p in ports}

        # One FFTW plan per port (avoid contention since three axes per call
        # are serialised inside publish() anyway). Plans get the MEASURE-grade
        # one-shot cost at first use, not at construction.
        self._plans = {p: _make_fft_plan(window_size) for p in ports}

        # Pre-computed axes for the client (constant per session, not per
        # snapshot — saves bandwidth). Raw spans the last hop in time;
        # FFT covers the first display_points rFFT bins (k=1..N) of the full
        # window, giving 3 Hz/bin up to ~768 Hz.
        self.time_axis_ms = np.linspace(0.0,
                                        1000.0 * self.raw_samples / sample_rate,
                                        display_points,
                                        dtype=np.float32).tolist()
        self.freq_axis_hz = (
            np.arange(1, display_points + 1, dtype=np.float32)
            * (sample_rate / window_size)
        ).tolist()

    def publish(self, port, window):
        """Called from the waveform worker thread on every window. Both raw
        and FFT compute on every call — no throttle. Each publish bumps the
        bus once, so the SSE pushes both freshly-computed views together."""
        try:
            raw_xyz = self._compute_raw(window)
        except Exception as e:
            print(f"[waveform] raw compute failed for {port}: {e}")
            return

        try:
            fft_xyz = self._compute_fft(port, window)
        except Exception as e:
            print(f"[waveform] fft compute failed for {port}: {e}")
            fft_xyz = None

        with self._lock:
            self._raw_seq[port] += 1
            entry = {
                "raw_seq": self._raw_seq[port],
                "raw": raw_xyz,
                "ts": time.time(),
                "fft_seq": self._fft_seq[port],
                "fft": (self._latest.get(port) or {}).get("fft"),
            }
            if fft_xyz is not None:
                self._fft_seq[port] += 1
                entry["fft_seq"] = self._fft_seq[port]
                entry["fft"] = fft_xyz
            self._latest[port] = entry

        if self._on_publish is not None:
            self._on_publish()

    def _compute_raw(self, window):
        """Decimated, smoothed view of the last `raw_samples` rows of window."""
        tail = window[-self.raw_samples:, :]
        out = {}
        for axis_idx, axis_name in enumerate(("x", "y", "z")):
            axis = np.ascontiguousarray(tail[:, axis_idx], dtype=np.float32)
            smoothed = _savgol(axis)
            out[axis_name] = _decimate_to(smoothed, self.display_points).tolist()
        return out

    def _compute_fft(self, port, window):
        """Magnitude of the first display_points rFFT bins (DC excluded) of
        the full 2604-sample window, per axis. Uses the cached FFTW plan."""
        plan = self._plans[port]
        out = {}
        for axis_idx, axis_name in enumerate(("x", "y", "z")):
            axis = np.ascontiguousarray(window[:, axis_idx], dtype=np.float32)
            mag = _run_fft_plan(plan, axis)
            out[axis_name] = mag[1:1 + self.display_points].astype(np.float32).tolist()
        return out

    def snapshot(self):
        """Build the SSE payload. Cheap — copies the cached per-port dicts."""
        with self._lock:
            ports_out = {}
            for port, entry in self._latest.items():
                if entry is None:
                    ports_out[port] = {
                        "raw_seq": 0,
                        "fft_seq": 0,
                        "ts": None,
                        "raw": None,
                        "fft": None,
                    }
                else:
                    ports_out[port] = entry
        return {
            "ports": ports_out,
            "time_axis_ms": self.time_axis_ms,
            "freq_axis_hz": self.freq_axis_hz,
            "fft_backend": FFT_BACKEND_NAME,
            "sample_rate": self.sample_rate,
            "window_size": self.window_size,
            "raw_samples": self.raw_samples,
            "now": time.time(),
        }
