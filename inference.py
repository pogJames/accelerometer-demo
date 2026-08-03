"""Worker 2 — frozen backbone + cosine-sim classifier head.

Each (WINDOW, 3) window → backbone → 128-d embedding → head.predict()
→ (class_id, similarity, class_name). Trained instantly from data/*.bin
(or legacy *.csv) via trainer.py.

This build targets the NXP i.MX Ethos-U NPU (Vela-compiled int8 model through
`tflite_runtime` + the Ethos-U delegate), and falls back to the float32 model
on CPU. To run on a different NPU (e.g. Matrix800 / VeriSilicon), change the
three PORT-marked spots below — see "Porting to another NPU" in README.md.
"untrained" and "stub" states let the dashboard render before artifacts exist.
"""

import os
import threading
import time
import numpy as np

from classifier import ClassifierHead, DEFAULT_HEAD_PATH


# ── Backend paths (NXP i.MX / Ethos-U) ────────────────────────────────────
# PORT: to target another NPU, change DELEGATE_PATH + NPU_MODEL_PATH here and
# the runtime import inside _try_load_interpreter(). README "Porting to another NPU".
NPU_MODEL_PATH = "models/vibration_backbone_int8_vela.tflite"  # Vela-compiled int8, Ethos-U only
CPU_MODEL_PATH = "models/vibration_backbone.tflite"            # float32, CPU fallback / FORCE_CPU=1
DELEGATE_PATH  = "/usr/lib/libethosu_delegate.so"             # PORT: Ethos-U delegate


def _resolve_default_model_path():
    """Best-guess for early display; the real one is set when a backend loads."""
    return NPU_MODEL_PATH if os.path.exists(NPU_MODEL_PATH) else CPU_MODEL_PATH


class InferenceWorker(threading.Thread):
    def __init__(self, window_queue, rolling_predictions,
                 model_path=None,
                 delegate_path=DELEGATE_PATH,
                 head_path=DEFAULT_HEAD_PATH):
        super().__init__(daemon=True)
        self.window_queue = window_queue
        self.rolling_predictions = rolling_predictions
        self.model_path = model_path or _resolve_default_model_path()
        self.delegate_path = delegate_path
        self.head_path = head_path

        # Classify only. The waveform path is fed by its own raw_queue and a
        # dedicated drain thread in app.py, so this worker no longer competes
        # with it — a single get→classify→push loop is enough.
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
        """Load the Ethos-U NPU backend, falling back to CPU then stub.
        Sets self.mode/_interp/_inp/_out on success.

        FORCE_CPU=1 skips the NPU delegate and runs the float model on CPU.

        PORT: to target another NPU, change the runtime import just below plus
        DELEGATE_PATH / NPU_MODEL_PATH above. See README "Porting to another NPU".
        """
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate   # PORT: runtime
        except ImportError as e:
            print(f"[inference] tflite_runtime not available ({e}); stub mode")
            return

        force_cpu = os.environ.get("FORCE_CPU") == "1"

        # NPU: Vela-compiled model through the Ethos-U delegate.
        if not force_cpu and os.path.exists(NPU_MODEL_PATH) and os.path.exists(self.delegate_path):
            try:
                delegate = load_delegate(self.delegate_path)
                interp = Interpreter(model_path=NPU_MODEL_PATH,
                                     experimental_delegates=[delegate])
                interp.allocate_tensors()
                self._commit(interp, NPU_MODEL_PATH, "npu")
                print(f"[inference] NPU delegate loaded from {self.delegate_path}")
                return
            except Exception as e:
                print(f"[inference] NPU delegate failed ({e}); falling back to CPU")

        # CPU: float model, no delegate (the Vela model is Ethos-U only, so a
        # CPU fallback must use the float artifact, not NPU_MODEL_PATH).
        if not os.path.exists(CPU_MODEL_PATH):
            print(f"[inference] CPU model not found at {CPU_MODEL_PATH}; stub mode")
            return
        try:
            interp = Interpreter(model_path=CPU_MODEL_PATH)
            interp.allocate_tensors()
            self._commit(interp, CPU_MODEL_PATH, "cpu")
            reason = "FORCE_CPU=1" if force_cpu else "no working NPU delegate"
            print(f"[inference] {reason}; CPU mode")
        except Exception as e:
            print(f"[inference] CPU load failed ({e}); stub mode")

    def _commit(self, interp, model_path, mode):
        """Latch a loaded interpreter as the active backend."""
        self._interp = interp
        self.model_path = model_path
        self.mode = mode
        self._inp = interp.get_input_details()[0]
        self._out = interp.get_output_details()[0]
        print(f"[inference] backbone={os.path.basename(model_path)} mode={mode} "
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

    def run(self):
        """Single classify loop: drain the cross-process window_queue, run the
        backbone+head, push the result. The waveform path lives entirely in
        app.py (raw_queue → drain thread → WaveformAggregator), so this worker
        doesn't fan out or compete with it."""
        self._try_load_interpreter()
        self.head = ClassifierHead.load(self.head_path)
        if self.head is None:
            print(f"[inference] no classifier head at {self.head_path}; serving 'untrained'")
        else:
            print(f"[inference] loaded head with {len(self.head.labels)} labels: "
                  f"{self.head.labels}")
        print("[inference] classify loop up")

        while not self.stopped():
            try:
                port, window = self.window_queue.get(timeout=0.5)
            except Exception:
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
