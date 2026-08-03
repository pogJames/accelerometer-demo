# Architecture

Per-module design rationale (the "why") lives in [notes.md](notes.md); this is
the data-flow overview.

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

W3  app.py + Flask       (serves the pages + the HTTP API)
     /                  live waveform (uPlot, raw scroll + FFT)
     /inference         live classification cards
     /metrics           per-sensor metric tables
     /record            data recording (delete inline)
     /train             train the classifier head
     /settings          per-port alias names
```

**Dual cadence in the reader.** `window_queue` carries full 2604-sample windows
for the model at ~6 Hz; `raw_queue` carries small 217-sample fresh chunks for
the waveform at ~36 Hz. Inference and waveform are fully decoupled — neither
blocks the other.

**Ring-buffer waveform.** `WaveformAggregator` keeps a 2 s circular sample
buffer per port. The display-tick thread reads the last N samples on every tick
(raw scrolls smoothly under a large overlapping window), and recomputes the FFT
at ~6 Hz from the last `WINDOW_SIZE` samples.

**Inference modes** (shown in sidebar):
- `npu` — Ethos-U delegate loaded, Vela int8 backbone on the NXP NPU (the only
  real backend).
- `stub` — runtime, model, or delegate missing → dashboard still renders. No CPU
  fallback: off-hardware always lands here.

For the process model, GIL/FIFO reasoning, and reader gating see
[notes.md](notes.md). For retargeting the NPU see notes.md → *Porting to another
NPU*.
