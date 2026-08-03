import os
import threading
import time
import numpy as np

from classifier import ClassifierHead, DEFAULT_HEAD_PATH


# PORT: same Matrix800 NPU, two drivers. Change these two + the runtime import
# in _try_load_interpreter to match the board's driver (NXP or Mesa). See
# docs/getting-started.md "Choose your NPU driver".
NPU_MODEL_PATH = "models/vibration_backbone_int8_vela.tflite"  # Vela int8, NXP/Ethos-U
DELEGATE_PATH  = "/usr/lib/libethosu_delegate.so"             # PORT


class InferenceWorker(threading.Thread):
    def __init__(self, window_queue, rolling_predictions,
                 model_path=NPU_MODEL_PATH,
                 delegate_path=DELEGATE_PATH,
                 head_path=DEFAULT_HEAD_PATH):
        super().__init__(daemon=True)
        self.window_queue = window_queue
        self.rolling_predictions = rolling_predictions
        self.model_path = model_path
        self.delegate_path = delegate_path
        self.head_path = head_path

        self._stopper = threading.Event()
        self.mode = "stub"      # "npu" | "stub"
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
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate   # PORT
        except ImportError as e:
            print(f"[inference] tflite_runtime not available ({e}); stub mode")
            return

        if not os.path.exists(NPU_MODEL_PATH):
            print(f"[inference] model not found at {NPU_MODEL_PATH}; stub mode")
            return
        if not os.path.exists(self.delegate_path):
            print(f"[inference] Ethos-U delegate not found at {self.delegate_path}; stub mode")
            return

        try:
            delegate = load_delegate(self.delegate_path)
            interp = Interpreter(model_path=NPU_MODEL_PATH,
                                 experimental_delegates=[delegate])
            interp.allocate_tensors()
        except Exception as e:
            print(f"[inference] NPU delegate failed ({e}); stub mode")
            return

        self._interp = interp
        self.mode = "npu"
        self._inp = interp.get_input_details()[0]
        self._out = interp.get_output_details()[0]
        print(f"[inference] NPU delegate loaded from {self.delegate_path}")
        print(f"[inference] backbone={os.path.basename(NPU_MODEL_PATH)} mode={self.mode} "
              f"input={self._inp['shape']} {self._inp['dtype']} "
              f"output={self._out['shape']} {self._out['dtype']}")
        print(f"[inference] input quant: {self._inp.get('quantization')}  "
              f"output quant: {self._out.get('quantization')}")

    def reload_head(self):
        new_head = ClassifierHead.load(self.head_path)
        with self._head_lock:
            self.head = new_head
        if new_head is None:
            print(f"[inference] head reload: missing or invalid at {self.head_path}")
        else:
            print(f"[inference] head reloaded: {len(new_head.labels)} labels "
                  f"{new_head.labels}")

    def embed(self, window: np.ndarray) -> np.ndarray:
        if self._interp is None:
            raise RuntimeError("interpreter not loaded")

        # CLIP-before-cast: a bare cast silently wraps out-of-range values
        # (200 -> -56), feeding the NPU garbage. See docs/architecture.md.
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
