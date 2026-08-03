# Files & tuning knobs

## Key files

| File | Role |
|---|---|
| `app.py` | Flask app, wires reader subprocesses + InferenceWorker + drains + display tick, hosts all SSE endpoints |
| `sensor_reader.py` | One reader subprocess per port: FC04 dual-cadence raw stream + FC03 metric batches |
| `fast_modbus.py` | Minimal Modbus RTU client — FC03 / FC04 / FC06 over pyserial, no pymodbus |
| `waveform.py` | `WaveformAggregator` — per-port ring buffer, `append()` + 30 Hz `render_tick()` (raw smoothed/decimated, FFT throttled ~6 Hz) |
| `inference.py` | `InferenceWorker` — single classify loop: window → backbone embed → head classify → rolling.push |
| `classifier.py` | `ClassifierHead` — cosine similarity, load/save JSON |
| `trainer.py` | `TrainerManager` — prototype computation, background thread |
| `recorder.py` | `RecordingManager` + `delete_label()` — two-stage queue, binary writer thread |
| `state.py` | `RollingPredictions`, `SnapshotBus`, `LatestSlot` |
| `port_aliases.json` | Friendly-name overrides — written by `/settings`, loaded into every template's context |
| `models/vibration_backbone_int8_vela.tflite` | Frozen int8 CNN backbone, Vela-compiled — the loaded NPU backbone (Ethos-U) |
| `models/vibration_backbone_int8.tflite` | Non-Vela int8 backbone — only for the VeriSilicon port (see notes.md → Porting) |
| `classifier_head.json` | Trained head — written by trainer, read by inference |
| `data/` | Recordings — `<label>.bin` float32 raw XYZ |

## Tuning knobs

| Constant | File | Default | Effect |
|---|---|---|---|
| `WINDOW_SIZE` | `sensor_reader.py` | 2604 | **Model** input window (samples). Matches the backbone — don't change without retraining. |
| `HOP_SIZE` | `sensor_reader.py` | 1302 | Window stride for inference (~6 emits/s). |
| `RAW_CHUNK_SIZE` | `sensor_reader.py` | 217 | Fresh-sample chunk size for the waveform path (~36 emits/s). |
| `WINDOW_SIZE` | `waveform.py` | 3906 | **FFT** window (samples). Independent from the model window — change for finer freq resolution. |
| `RING_CAPACITY` | `waveform.py` | `SAMPLE_RATE*2` | Per-port circular sample buffer (~2 s). Must be ≥ max FFT/raw window. |
| `FFT_INTERVAL_S` | `waveform.py` | `1/4` | FFT recompute interval at the display tick. |
| `KURT_POLL_INTERVAL_S` | `sensor_reader.py` | 0.4 | Kurtosis change-detect poll cadence. |
| `METRIC_FALLBACK_S` | `sensor_reader.py` | 5.0 | Force a metric batch emit if kurtosis sits perfectly still. |
| `METRIC_READ_TIMEOUT_S` | `sensor_reader.py` | 5.0 | Serial read timeout for FC03 (matches vendor's Rust reference). |
| `ROLLING_WINDOW` | `state.py` | 7 | Predictions in the rolling majority-vote buffer. |
| `DISPLAY_REFRESH_EVERY` | `state.py` | 2 | Pushes between inference dashboard refreshes. |
| `TRAIN_HOP` | `trainer.py` | 651 | Training window stride (75 % overlap). |
| `MIN_WINDOWS` | `trainer.py` | 6 | Minimum windows per label to allow training. |
