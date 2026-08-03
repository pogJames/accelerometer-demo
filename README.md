# npu-classifier

**English** · [繁體中文](README.zh-TW.md)

Real-time vibration monitor + classifier. A tri-axial accelerometer streams raw XYZ samples over Modbus RTU → a frozen CNN backbone extracts 128-d embeddings → a cosine-similarity head classifies motion → live waveform, FFT spectrum, sensor metrics and inference results all surface in a Flask dashboard. No retraining the backbone; the head is computed in seconds from your own recordings.


## Hardware setup

| Item | Detail |
|---|---|
| Sensor | Tri-axial accelerometer, RS485/Modbus RTU |
| Interface | USB–RS485 adapter → `/dev/ttyUSB0..3` (up to 4 sensors) |
| Baud rate | **3 Mbps** (required for raw streaming) |
| Sample rate | 7812 Hz nominal (written to sensor register `0x01` on connect) |
| Modbus FC04 | register `0x02` — FIFO count + raw XYZ samples |
| Modbus FC03 | computed metrics (RMS, peak, kurtosis, …) — see `sensor_reader.py` register map |


## Setup & launch

```bash
# 1. Create venv and install deps
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Run
python app.py                   # http://localhost (port 80)
FORCE_CPU=1 python app.py       # bypass NPU delegate (debug / dev machine)
```

The app auto-detects serial ports listed in `ALLOWED_PORTS` (`/dev/ttyUSB0..3`). One reader subprocess is spawned per detected sensor.


## Architecture

```
W1  sensor_reader.py             one subprocess per port
     ────────────────────────
     FC04 register 0x02   → decode → /8192 → G
       │                         ├─→ recorder.feed()          (local, no IPC)
       │                         ├─→ accumulate 2604-sample window  ──┐
       │                         │   (hop=1302, ~6 Hz/port)           │
       │                         └─→ accumulate 217-sample chunk  ──┐ │
       │                             (~36 Hz/port)                  │ │
       │                                                            │ │
     FC03 metric batch  (kurtosis change-detect, ~2-5 s) ─────────┐  │ │
                                                                 │  │ │
                                                                 │  │ │
                                                 metrics_queue ──┘  │ │
                                                    window_queue ───┘ │
                                                        raw_queue ────┘
                                                          │
                                  ────── main process ────┴──────
                                       ↓               ↓                 ↓
                            InferenceWorker     raw drain →       metrics drain →
                            (classify)         WaveformAggregator   per-port
                              ↓                .append() to ring     latest dict
                            RollingPreds       ↓                     ↓
                              ↓                display-tick 30 Hz   MetricsBus
                            SnapshotBus         render_tick()
                                                ↓
                                               WaveformBus

W3  app.py + Flask       (serves the pages below + the HTTP API — see § HTTP API)
     /                  live waveform (uPlot, raw scroll + FFT)
     /inference         live classification cards
     /metrics           per-sensor metric tables
     /record            data recording (delete inline)
     /train             train the classifier head
     /settings          per-port alias names
```

**Dual cadence in the reader.** `window_queue` carries full 2604-sample windows for the model at ~6 Hz; `raw_queue` carries small 217-sample fresh chunks for the waveform at ~36 Hz. Inference and waveform are fully decoupled — neither blocks the other.

**Ring-buffer waveform.** `WaveformAggregator` keeps a 2 s circular sample buffer per port. The display-tick thread reads the last N samples on every tick (raw scrolls smoothly under a large overlapping window), and recomputes the FFT at ~6 Hz from the last `WINDOW_SIZE` samples.

**Inference modes** (shown in sidebar):
- `npu` — `libteflon.so` delegate loaded, int8 backbone on VeriSilicon NPU
- `cpu` — `ai_edge_litert` without delegate
- `stub` — no model file or no `ai_edge_litert`; dashboard still renders


## HTTP API

All bodies are JSON. Errors return `{"error": "..."}` with a 4xx/5xx status. Ports are the raw `/dev/ttyUSBx` strings.

### SSE streams

Long-lived `text/event-stream` connections; each `data:` frame is one JSON snapshot, pushed on change (heartbeat comment on idle). A stream is inert until a page activates the relevant port — see [Pages](#pages).

| Endpoint | Push trigger | Frame payload |
|---|---|---|
| `GET /stream/inference` | new majority-vote latch | `{ports:{<port>:{majority_class_name, majority_count, window_count, recent[], display_seq, latest_ts}}, class_labels[], label_colors{}, active_ports[], inference_mode, now}` |
| `GET /stream/waveform` | 30 Hz display tick | `{ports:{<port>:{raw, fft, raw_seq, fft_seq, ts}}, raw_axis[], freq_axis_hz[], sample_rate, window_size, fft_bins, …}` |
| `GET /stream/metrics` | FC03 batch (~2–5 s) | `{ports:{<port>:{temperature, gravity{rms,peak,crest,skewness,kurtosis,primary_freq}, velocity{rms,peak,crest,primary_freq}, ts}}, open_ports[], now}` |

### Activation & config (POST)

| Endpoint | Request body | Response | Effect |
|---|---|---|---|
| `POST /api/active_port` | `{port, active: bool}` | `{port, active}` | Open/close a port for **raw + inference** streaming |
| `POST /api/metrics_active` | `{port, active: bool}` | `{port, active}` | Open/close **FC03 metric** polling (independent of raw) |
| `POST /api/waveform_config` | `{fft_max_hz?, raw_samples?}` | applied values (`{fft_max_hz, fft_bins, raw_samples}`) | Live waveform/FFT knobs |
| `POST /api/port_alias` | `{port, alias}` | `{port, alias}` | Set a port's friendly display name |

### Recording

| Endpoint | Request body | Response |
|---|---|---|
| `GET /api/recordings` | — | `{labels:[{name, samples, windows, eligible}], ports[], sample_rate, min_samples, min_windows, window_size, data_dir}` |
| `GET /api/record/status` | — | `{session: {status, name, port, progress, elapsed_s, samples_written, target_samples, file_path} \| null}` (client-polled ~1 Hz) |
| `POST /api/record/start` | `{name, target_samples, port, mode:"append"\|"overwrite"}` | `{session}` |
| `POST /api/record/cancel` | — | `{session}` (keeps samples written so far) |
| `POST /api/recordings/delete` | `{name}` | `{deleted}` — `409` if that label is recording |

### Training

| Endpoint | Request body | Response |
|---|---|---|
| `GET /api/train/status` | — | `{session, head_labels[], head_colors{}, backbone, backbone_mode}` |
| `POST /api/train/start` | `{labels:[...]}` (≥ 2 eligible) | `{session}` |
| `POST /api/train/cancel` | — | `{session}` |

### One-shot snapshots (JSON)

Non-streaming GETs returning the same payload as the matching `/stream/*` feed — for `curl`/`jq` debugging without holding an SSE connection open. Reflect the current cached state regardless of whether any port is activated.

| Endpoint | Response |
|---|---|
| `GET /api/inference` | Current `/stream/inference` payload |
| `GET /api/metrics` | Current `/stream/metrics` payload |
| `GET /api/waveform` | Current `/stream/waveform` payload |


## Pages

### Live waveform `/`

2 × 2 grid of per-sensor uPlot canvases.

- **Raw mode** — scrolling time-domain trace, X/Y/Z overlaid, "Latest N samples" configurable up to 1 s, Refresh rate selectable 10/30/60 fps.
- **FFT mode** — magnitude spectrum, FFT range configurable in Hz (caps at 800 Hz).
- Per-axis stat chips above each plot (min..max in raw, peak frequency in FFT). Click a chip to hide/show that axis.
- Hover the plot for an inline tooltip with the sample value at that point.

### Live inference `/inference`

One accordion card per sensor. Shows majority class label over the last 7 results, latest confidence, and a colour-coded dot strip. SSE-driven, no polling.

### Metrics data `/metrics`

Per-sensor card showing temperature + gravity table (RMS / Peak / Crest / Skewness / Kurtosis × XYZ + primary freq) + velocity table (RMS / Peak / Crest × XYZ + primary freq). The reader polls kurtosis at ~2 Hz and emits a fresh **batch** of all metrics only when kurtosis changes (or 5 s fallback) — so every field on the page advances together at the slow ~2–5 s cadence.

Activating `/metrics` does **not** start raw streaming on a port — FC03 polling only. The `/` and `/inference` pages activate raw separately; raw streaming always preempts metric polling on the same port.

### Data recording `/record`

Records raw XYZ samples to `data/<name>.bin` (float32 little-endian, 3 channels interleaved). Each row in the "Existing data" table has a trash-icon delete button (confirmation prompt). Append / Overwrite modes; progress streams over SSE.

> To convert a recorded `.bin` to `.csv` (`x,y,z`, no timestamps), run the converter manually: `python bin2csv.py data/<name>.bin` (also accepts a glob or a directory).

### Train model `/train`

"Training" = computing one mean embedding per label. No gradients, takes seconds. Select ≥ 2 eligible labels → click **Train model** → `classifier_head.json` written → `InferenceWorker` hot-reloads with no app restart. Colour slots are randomly re-assigned on each retrain.

### Settings `/settings`

One row per detected port; type a friendly alias (e.g. "Front-left motor") and it's saved instantly to `port_aliases.json`. Aliases render across every page that shows a port name; the raw `/dev/ttyUSBx` paths remain the internal IDs.


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
| `vibration_backbone_int8.tflite` | Frozen int8 CNN backbone (NPU target) |
| `classifier_head.json` | Trained head — written by trainer, read by inference |
| `data/` | Recordings — `<label>.bin` float32 raw XYZ |


## Tuning knobs

| Constant | File | Default | Effect |
|---|---|---|---|
| `WINDOW_SIZE` | `sensor_reader.py` | 2604 | **Model** input window (samples). Matches the backbone — don't change without retraining. |
| `HOP_SIZE` | `sensor_reader.py` | 1302 | Window stride for inference (~6 emits/s). |
| `RAW_CHUNK_SIZE` | `sensor_reader.py` | 217 | Fresh-sample chunk size for the waveform path (~36 emits/s). |
| `WINDOW_SIZE` | `waveform.py` | 2604 | **FFT** window (samples). Independent from the model window — change for finer freq resolution. |
| `RING_CAPACITY` | `waveform.py` | `SAMPLE_RATE*2` | Per-port circular sample buffer (~2 s). Must be ≥ max FFT/raw window. |
| `FFT_INTERVAL_S` | `waveform.py` | `1/6` | FFT recompute interval at the display tick. |
| `KURT_POLL_INTERVAL_S` | `sensor_reader.py` | 0.4 | Kurtosis change-detect poll cadence. |
| `METRIC_FALLBACK_S` | `sensor_reader.py` | 5.0 | Force a metric batch emit if kurtosis sits perfectly still. |
| `METRIC_READ_TIMEOUT_S` | `sensor_reader.py` | 5.0 | Serial read timeout for FC03 (matches vendor's Rust reference). |
| `ROLLING_WINDOW` | `state.py` | 7 | Predictions in the rolling majority-vote buffer. |
| `DISPLAY_REFRESH_EVERY` | `state.py` | 2 | Pushes between inference dashboard refreshes. |
| `TRAIN_HOP` | `trainer.py` | 651 | Training window stride (75 % overlap). |
| `MIN_WINDOWS` | `trainer.py` | 6 | Minimum windows per label to allow training. |
