# Architecture

How the system is put together and why. For per-file notes see
[modules.md](modules.md); for the HTTP surface see [api.md](api.md).

---

## Three workers

The pipeline is three workers, decoupled by queues.

```
W1  sensor_reader.py             one subprocess per port
     FC04 register 0x02 → decode → /8192 → G
       ├─→ recorder.feed()               (local, no IPC)
       ├─→ 2604-sample window  ─┐  hop 1302, ~6 Hz/port  ──→ window_queue ─┐
       └─→ 217-sample chunk  ─┐ │  ~36 Hz/port            ──→ raw_queue ──┐ │
     FC03 metric batch ─┐     │ │  (kurtosis change, ~2–5 s) → metrics_queue │ │
                        │     │ │                                        │  │ │
    ──── main process ──┴─────┴─┴────────────────────────────────────────┴──┴─┘
        InferenceWorker      raw drain →           metrics drain →
        (classify)           WaveformAggregator    per-port latest
          ↓                    ↓ 30 Hz tick          ↓
        RollingPredictions   WaveformBus           MetricsBus
          ↓
        SnapshotBus

W3  app.py — Flask, wires it all, serves pages + SSE
```

- **W1 — `sensor_reader.py`.** One subprocess per port. Reads Modbus, emits
  `(WINDOW_SIZE, 3)` windows for inference and small raw chunks for the
  waveform. It runs in its **own process** because the serial poll loop is
  GIL-bound: sharing the GIL with Flask, inference, or GC pauses lets the
  sensor's hardware FIFO overflow and drop samples. The subprocess also owns the
  recorder; raw samples never cross the process boundary. Control (start / cancel
  / status) is RPC over an `mp.Queue` pair.
- **W2 — `inference.py`.** One thread. Runs the frozen backbone plus the
  cosine-similarity head. Classify only — the waveform has its own queue and
  drain thread, so W2 never competes with it.
- **W3 — `app.py`.** Wires everything and serves the dashboard over SSE.

### Dual cadence

The reader emits at two rates, fully decoupled:

- `window_queue` — full 2604-sample windows for the model, ~6 Hz.
- `raw_queue` — small 217-sample fresh chunks for the waveform, ~36 Hz.

Inference and the waveform never block each other.

### Waveform ring buffer

`WaveformAggregator` keeps a ~2 s circular buffer per port. The 30 Hz display
tick reads the last N samples every tick (raw scrolls smoothly), and recomputes
the FFT ~6 Hz from the last FFT-window samples.

---

## NPU backend

Two modes, shown in the sidebar:

- **`npu`** — driver, delegate, and int8 backbone all loaded. The only real
  backend.
- **`stub`** — runtime, model, or delegate missing, or the delegate failed to
  load. Returns a fixed class so the dashboard still renders. A dev machine
  always lands here.

There is **no CPU fallback** in this build.

### The silent-garbage trap

A float32 `.tflite`, or the wrong delegate/model for the board's driver, **does
not error and does not warn**. It loads, runs, and returns bit-identical output
for every input. Two rules avoid it:

1. Match the runtime, delegate, and model to the board's NPU driver. See
   [getting-started → step 3](getting-started.md#3-choose-your-npu-driver).
2. Feed the model int8 correctly. **Clip before cast** when quantizing input:
   `np.clip(np.round(x/scale + zp), -128, 127).astype(np.int8)`. A bare
   `.astype` wraps out-of-range values (200 → −56) and feeds the NPU garbage.

The backbone already ends in `Dense(N, softmax)` — the output is the probability
vector. Do not apply softmax again in the classify path.

### Same NPU, two drivers

The Matrix800 has one NPU. Its Linux image uses one of two drivers, and each
needs a different runtime, delegate, and model file. All three knobs are in
`inference.py`, marked `# PORT`. The full table is in
[getting-started → step 3](getting-started.md#3-choose-your-npu-driver).

---

## Files

| File | Role |
|---|---|
| `app.py` | Flask app; wires reader subprocesses + InferenceWorker + drains + display tick; hosts all SSE endpoints |
| `sensor_reader.py` | Reader subprocess per port: FC04 dual-cadence raw stream + FC03 metric batches |
| `fast_modbus.py` | Minimal Modbus RTU client (FC03 / FC04 / FC06 over pyserial, no pymodbus) |
| `waveform.py` | `WaveformAggregator` — per-port ring buffer, 30 Hz render tick |
| `inference.py` | `InferenceWorker` — window → backbone embed → head classify → push |
| `classifier.py` | `ClassifierHead` — cosine similarity, JSON load/save |
| `trainer.py` | `TrainerManager` — prototype computation, background thread |
| `recorder.py` | `RecordingManager` + `delete_label()` — two-stage queue, binary writer thread |
| `state.py` | `RollingPredictions`, `SnapshotBus`, `LatestSlot` |
| `bin2csv.py` | Convert `.bin` recordings to CSV |
| `port_aliases.json` | Friendly-name overrides, written by `/settings` |
| `models/vibration_backbone_int8_vela.tflite` | Vela-compiled int8 backbone — NXP driver (Ethos-U) |
| `models/vibration_backbone_int8.tflite` | Non-Vela int8 backbone — Mesa driver |
| `classifier_head.json` | Trained head — written by trainer, read by inference |
| `data/` | Recordings — `<label>.bin`, float32 raw XYZ |

Per-file design notes: [modules.md](modules.md).

---

## Tuning knobs

| Constant | File | Default | Effect |
|---|---|---|---|
| `WINDOW_SIZE` | `sensor_reader.py` | 2604 | **Model** input window (samples). Matches the backbone — don't change without retraining. |
| `HOP_SIZE` | `sensor_reader.py` | 1302 | Window stride for inference (~6 emits/s). |
| `RAW_CHUNK_SIZE` | `sensor_reader.py` | 217 | Fresh-sample chunk for the waveform (~36 emits/s). |
| `WINDOW_SIZE` | `waveform.py` | 3906 | **FFT** window (samples). Independent from the model window — change for finer frequency resolution. |
| `RING_CAPACITY` | `waveform.py` | `SAMPLE_RATE*2` | Per-port circular buffer (~2 s). Must be ≥ the largest FFT/raw window. |
| `FFT_INTERVAL_S` | `waveform.py` | `1/4` | FFT recompute interval at the display tick. |
| `KURT_POLL_INTERVAL_S` | `sensor_reader.py` | 0.4 | Kurtosis change-detect poll cadence. |
| `METRIC_FALLBACK_S` | `sensor_reader.py` | 5.0 | Force a metric emit if kurtosis sits still. |
| `METRIC_READ_TIMEOUT_S` | `sensor_reader.py` | 5.0 | Serial read timeout for FC03. |
| `ROLLING_WINDOW` | `state.py` | 7 | Predictions in the majority-vote buffer. |
| `DISPLAY_REFRESH_EVERY` | `state.py` | 2 | Pushes between inference dashboard refreshes. |
| `TRAIN_HOP` | `trainer.py` | 651 | Training window stride (75 % overlap). |
| `MIN_WINDOWS` | `trainer.py` | 6 | Minimum windows per label to allow training. |

> `WINDOW_SIZE` exists in **both** `sensor_reader.py` (model input) and
> `waveform.py` (FFT window). They are independent — do not unify them.
