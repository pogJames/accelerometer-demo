"""Worker 2 — frozen backbone + cosine-sim classifier head.

Each (WINDOW, 3) window → backbone → 128-d embedding → head.predict()
→ (class_id, similarity, class_name). Trained instantly from data/*.bin
(or legacy *.csv) via trainer.py. Backbone variant `vibration_backbone_int8.tflite` is
preferred for NPU; `vibration_backbone.tflite` (float32) is the CPU
fallback — _classify() handles both via dtype-aware quant/dequant.
"untrained" and "stub" states let the dashboard render before artifacts exist.
"""

import os
import queue
import threading
import time
import numpy as np

from classifier import ClassifierHead, DEFAULT_HEAD_PATH


INT8_BACKBONE_PATH    = "vibration_backbone_int8.tflite"
FLOAT_BACKBONE_PATH   = "vibration_backbone.tflite"
DEFAULT_DELEGATE_PATH = "/usr/local/lib/aarch64-linux-gnu/libteflon.so"


def _resolve_default_model_path():
    if os.path.exists(INT8_BACKBONE_PATH):
        return INT8_BACKBONE_PATH
    return FLOAT_BACKBONE_PATH


class InferenceWorker(threading.Thread):
    def __init__(self, window_queue, rolling_predictions,
                 model_path=None,
                 delegate_path=DEFAULT_DELEGATE_PATH,
                 head_path=DEFAULT_HEAD_PATH,
                 waveform_aggregator=None):
        super().__init__(daemon=True)
        self.window_queue = window_queue
        self.rolling_predictions = rolling_predictions
        self.model_path = model_path or _resolve_default_model_path()
        self.delegate_path = delegate_path
        self.head_path = head_path
        self._waveform = waveform_aggregator

        # In-process fan-out queues. The drainer (run()) keeps the upstream
        # cross-process mp.Queue empty by doing only get + put_nowait — no
        # GIL-heavy work. Downstream workers consume from their own queues
        # at their own pace; drop-oldest on Full means a slow downstream
        # never back-pressures the drainer. maxsize=8 ≈ 1.3 s of windows at
        # 6 Hz × 4 ports.
        self._classify_q = queue.Queue(maxsize=8)
        self._waveform_q = (queue.Queue(maxsize=8)
                            if waveform_aggregator is not None else None)
        self._classify_thread = None
        self._waveform_thread = None

        self._stopper = threading.Event()
        self.mode = "stub"      # "npu" | "cpu" | "stub"
        self._interp = None
        self._inp = None
        self._out = None
        # Wraps every set_tensor/invoke/get_tensor triple so trainer.py can
        # borrow this interpreter without racing the live inference loop.
        self._invoke_lock = threading.Lock()
        self._head_lock = threading.Lock()
        self.head = None
        self._debug_n = 0

    def stopIt(self):
        self._stopper.set()

    def stopped(self):
        return self._stopper.is_set()

    def _try_load_interpreter(self):
        """Try NPU → CPU → stub. Sets self.mode/_interp/_inp/_out on success.

        Set env var FORCE_CPU=1 to bypass the NPU delegate even when the
        libteflon.so file is present.
        """
        try:
            from ai_edge_litert.interpreter import Interpreter, load_delegate
        except ImportError as e:
            print(f"[inference] ai_edge_litert not available ({e}); stub mode")
            return

        if not os.path.exists(self.model_path):
            print(f"[inference] backbone not found at {self.model_path}; stub mode")
            return

        force_cpu = os.environ.get("FORCE_CPU") == "1"

        if os.path.exists(self.delegate_path) and not force_cpu:
            try:
                delegate = load_delegate(self.delegate_path)
                self._interp = Interpreter(model_path=self.model_path,
                                           experimental_delegates=[delegate])
                self.mode = "npu"
                print(f"[inference] NPU delegate loaded from {self.delegate_path}")
            except Exception as e:
                print(f"[inference] NPU delegate failed ({e}); falling back to CPU")
                self._interp = Interpreter(model_path=self.model_path)
                self.mode = "cpu"
        else:
            self._interp = Interpreter(model_path=self.model_path)
            self.mode = "cpu"
            reason = "FORCE_CPU=1" if force_cpu else f"no delegate at {self.delegate_path}"
            print(f"[inference] {reason}; CPU mode")

        self._interp.allocate_tensors()
        self._inp = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        print(f"[inference] backbone={os.path.basename(self.model_path)} mode={self.mode} "
              f"input={self._inp['shape']} {self._inp['dtype']} "
              f"output={self._out['shape']} {self._out['dtype']}")
        print(f"[inference] input quant: {self._inp.get('quantization')}  "
              f"output quant: {self._out.get('quantization')}")

    def reload_head(self):
        """Atomically swap the classifier head from disk. Called by trainer.py
        on completion so the live dashboard picks up the new labels without
        a process restart."""
        new_head = ClassifierHead.load(self.head_path)
        with self._head_lock:
            self.head = new_head
        if new_head is None:
            print(f"[inference] head reload: missing or invalid at {self.head_path}")
        else:
            print(f"[inference] head reloaded: {len(new_head.labels)} labels "
                  f"{new_head.labels}")

    def embed(self, window: np.ndarray) -> np.ndarray:
        """Run the backbone on one (WINDOW, 3) window. Returns 128-d float32.

        Public so `trainer.py` can borrow this interpreter to compute prototypes.
        Holds `_invoke_lock` for the duration of the TFLite call.
        """
        if self._interp is None:
            raise RuntimeError("interpreter not loaded")

        # Quantize input if model expects int8. CLIP-before-cast — a bare cast
        # silently wraps out-of-range values (200 -> -56), feeding the NPU
        # garbage. Documented in CLAUDE.md.
        if self._inp["dtype"] == np.int8:
            scale, zero_point = self._inp["quantization"]
            x = np.clip(np.round(window / scale + zero_point), -128, 127).astype(np.int8)
        elif self._inp["dtype"] == np.uint8:
            scale, zero_point = self._inp["quantization"]
            x = np.clip(np.round(window / scale + zero_point), 0, 255).astype(np.uint8)
        else:
            x = window.astype(self._inp["dtype"])

        x = x.reshape(self._inp["shape"])

        with self._invoke_lock:
            self._interp.set_tensor(self._inp["index"], x)
            self._interp.invoke()
            y_raw = self._interp.get_tensor(self._out["index"])

        if self._out["dtype"] in (np.int8, np.uint8):
            o_scale, o_zero_point = self._out["quantization"]
            feat = (y_raw.astype(np.float32) - o_zero_point) * o_scale
        else:
            feat = y_raw.astype(np.float32)

        return feat.reshape(-1)

    def _classify(self, window: np.ndarray):
        """window: (WINDOW, 3) float32. Returns (class_id, confidence, class_name)."""
        if self.mode == "stub":
            return 0, 1.0, "stub"

        with self._head_lock:
            head = self.head
        if head is None:
            return 0, 0.0, "untrained"

        feat = self.embed(window)
        class_id, sim = head.predict(feat)
        class_name = head.labels[class_id] if 0 <= class_id < len(head.labels) else f"class_{class_id}"

        self._debug_n += 1
        if self._debug_n % 50 == 0:
            print(f"[inference #{self._debug_n}] window: "
                  f"min={float(window.min()):.3f} max={float(window.max()):.3f} "
                  f"std={float(window.std()):.3f} | "
                  f"feat: norm={float(np.linalg.norm(feat)):.3f} "
                  f"min={float(feat.min()):.3f} max={float(feat.max()):.3f} | "
                  f"sim={sim:.3f} -> {class_name}")

        return class_id, sim, class_name

    def _put_drop_oldest(self, q, item):
        """Non-blocking put with drop-oldest on Full. Used by the drainer to
        fan out into downstream in-process queues without ever back-pressuring
        the cross-process mp.Queue."""
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def _classify_loop(self):
        """Downstream classify worker. Reads from _classify_q at whatever
        rate _classify can sustain; if it falls behind, drop-oldest at the
        drainer keeps the freshest window in flight — the rolling-7 majority
        vote tolerates the missed windows."""
        while not self.stopped():
            try:
                port, window = self._classify_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                class_id, confidence, class_name = self._classify(window)
            except Exception as e:
                print(f"[inference] invoke failed for {port}: {e}")
                continue
            rolling = self.rolling_predictions.get(port)
            if rolling is None:
                continue
            rolling.push({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "ts": time.time(),
            })

    def _waveform_loop(self):
        """Downstream waveform worker. Reads from _waveform_q and runs the
        aggregator's savgol/FFT compute. numpy + pyfftw release the GIL, so
        this runs concurrently with the classify thread and the drainer."""
        while not self.stopped():
            try:
                port, window = self._waveform_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._waveform.publish(port, window)
            except Exception as e:
                print(f"[inference] waveform publish failed for {port}: {e}")

    def run(self):
        """DRAINER. Does only `get + 2× put_nowait` so it can't be GIL-starved
        by sibling threads doing heavy compute or JSON encoding. The cross-
        process mp.Queue stays drained regardless of downstream worker speed."""
        self._try_load_interpreter()
        self.head = ClassifierHead.load(self.head_path)
        if self.head is None:
            print(f"[inference] no classifier head at {self.head_path}; serving 'untrained'")
        else:
            print(f"[inference] loaded head with {len(self.head.labels)} labels: "
                  f"{self.head.labels}")

        # Spawn downstream workers AFTER interpreter load so the classify
        # thread doesn't race the _interp=None startup window.
        self._classify_thread = threading.Thread(
            target=self._classify_loop, daemon=True, name="classify-worker")
        self._classify_thread.start()
        if self._waveform is not None:
            self._waveform_thread = threading.Thread(
                target=self._waveform_loop, daemon=True, name="waveform-worker")
            self._waveform_thread.start()
        print(f"[inference] drainer + classify"
              f"{' + waveform' if self._waveform is not None else ''} threads up")

        while not self.stopped():
            try:
                port, window = self.window_queue.get(timeout=0.5)
            except Exception:
                continue
            # Fan out to whoever's downstream. Drop-oldest on each in-process
            # queue absorbs transient bursts without back-pressuring here.
            self._put_drop_oldest(self._classify_q, (port, window))
            if self._waveform_q is not None:
                self._put_drop_oldest(self._waveform_q, (port, window))
