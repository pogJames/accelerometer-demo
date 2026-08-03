# Modules

**English** · [繁體中文](modules.zh-TW.md)

Per-file design notes for anyone editing the internals. For the system view see
[architecture.md](architecture.md).

---

## app.py

Worker 3 — the Flask dashboard and the entry point that wires W1/W2/W3. Runs a
reader subprocess per port (each owns a local `RecordingManager`, reached from
the main process via `RemoteRecorder` RPC), an `InferenceWorker`, and a
`TrainerManager` that borrows the worker's interpreter to compute prototypes,
then hot-reloads the head. Run: `python app.py` → `http://<board-ip>/` (binds
`0.0.0.0:80`).

- The `site-packages` path insert lets an embedded build ship bundled deps.
  Harmless when empty. On Linux (fork) children inherit `sys.path`; on Windows
  (spawn) `reader_process_main` re-applies it.
- Queue caps: window pickling is ~1–2 ms for a `(2604,3)` window vs the 167 ms
  hop, so cap 16 gives 4 readers headroom before `put()` blocks; drop-oldest
  kicks in first. Metrics cap is small (~2–5 s emits); raw-chunk cap is 64.
- Reader gating: a reader runs if the user opened the port for inference **or** a
  recording is running. `inference_open` / `recording_ports` / `metrics_open`
  are tracked separately; raw (`active_events`) is the union of the first two,
  FC03 (`metrics_events`) tracks the third — so `/metrics` can poll without the
  3 Mbps raw cost. Recording on a port not opened for inference must not leave it
  "open" afterwards, so `begin_recording` / `end_recording` don't touch
  `inference_open`.
- `record_state["port"]` is set on start and never cleared, so `/status` keeps
  reporting the right subprocess's `last_finished`.
- Recording auto-stops in the subprocess at the target; the main process learns
  via a status poll (`release_if_finished`).
- The `/api/*` GETs are non-streaming siblings of the `/stream/*` feeds, for
  curl/jq without holding a connection open.

## sensor_reader.py

Streams raw XYZ from input register `0x02` (FC04) and emits `WINDOW_SIZE` sliding
windows (`HOP_SIZE` hop) to the inference queue. Read pattern mirrors the
validated Rust client; queues use drop-oldest.

`reader_process_main` arguments:
- `port` — e.g. `/dev/ttyUSB0`.
- `window_queue` — pushes `(port, ndarray)` for inference.
- `req_q` / `resp_q` — RPC in / responses out.
- `stop_event` — request shutdown.
- `active_event` — set when the port should read raw. When clear, the reader
  sleeps and sends no Modbus traffic. Each inactive→active transition re-writes
  the sample-rate register, which the sensor treats as a FIFO reset, so the first
  window after resume is fresh.
- `data_dir` — where recordings are written.

Notes:
- `spawn` gives the child a fresh interpreter, so it re-applies the
  `site-packages` path before importing project modules.
- No CPU pinning; the kernel schedules readers across cores. `SCHED_FIFO`
  priority 50 preempts normal processes when a reader has work (needs root /
  `CAP_SYS_NICE`; logged and skipped otherwise).
- `READ_TIMEOUT_S = 0.05`: 50 ms is well above the sensor's response time at
  3 Mbaud but turns a bad-packet stall into a ~100 ms blip. If `total_fails`
  climbs in normal use, raise it.
- **FC03 metric registers alias.** Each metric is its own
  `read_holding_registers(base, count=3)` — a contiguous block read returns
  garbage. Addresses and scaling mirror the Rust client (`src/types.rs`,
  `src/modbus.rs`).
- Metric polling: kurtosis (the slowest metric) is polled at
  `KURT_POLL_INTERVAL_S`; a full batch is emitted only when it changes (or first
  poll, or after `METRIC_FALLBACK_S`), so the whole set updates together at the
  slow 2–5 s cadence. FC03 reads are slow and variable (~0.5–1 s); a premature
  timeout leaves the slow reply in flight and the next request reads that stale
  reply, desyncing the stream. Hence `METRIC_READ_TIMEOUT_S = 5.0` plus a line
  drain, swapped in only around metric reads. Raw streaming always wins:
  `_MetricsAbort` bails a slow sweep the moment raw is requested.
- `skewness` can be negative on the wire — currently read unsigned; if it reads
  wrong, view as int16 (flagged for hardware validation).

## fast_modbus.py

Minimal Modbus RTU client over pyserial — FC03 / FC04 / FC06 only, no pymodbus.
Enough for this sensor and nothing else.

## waveform.py

Per-port live-waveform aggregator. Fed by the fast raw stream (sensor_reader
~217-sample chunks at ~36 Hz/port → app.py drain thread → `append(port, chunk)`)
into a per-port circular buffer. A ~30 Hz display tick calls `render_tick()`,
which smooths and decimates the raw view every tick (smooth scroll) and
recomputes the FFT every `FFT_INTERVAL_S` (~6 Hz). `snapshot()` is a cheap read
for the SSE handlers.

- FFT uses pyfftw with cached plans (NEON SIMD on aarch64), falling back to
  `numpy.fft.rfft` if pyfftw is absent.
- `WINDOW_SIZE` here is the **FFT window only** — independent from the model
  input window (`WINDOW_SIZE` in sensor_reader.py). Do not unify them.
- Savitzky-Golay coefficients (`_savgol_coeffs`) are derived with no scipy:
  build the Vandermonde matrix `A[i,k] = i**k` for `i in [-half, +half]`, take
  row 0 of `pinv(A)`. Equivalent to `scipy.signal.savgol_coeffs(W, P)`. `_savgol`
  pads edge-replicated so the ends aren't pulled toward zero.
- `WaveformBus` is separate from `state.SnapshotBus` so prediction and waveform
  SSE don't false-wake each other.

## inference.py

Each `(WINDOW, 3)` window → backbone → 128-d embedding → `head.predict()` →
`(class_id, similarity, class_name)`. Backend is the NPU only (see
[architecture → NPU backend](architecture.md#npu-backend)). If the runtime,
model, or delegate is missing, it drops to `stub` (class 0, confidence 1.0). No
CPU fallback.

- `embed()` is public so trainer.py can borrow the interpreter to compute
  prototypes; it holds `_invoke_lock` for the TFLite call.
- `reload_head()` atomically swaps the head from disk so a retrain surfaces new
  labels without a restart.
- **Clip before cast** when quantizing input:
  `np.clip(np.round(x/scale + zp), -128, 127).astype(np.int8)`. A bare `.astype`
  wraps out-of-range values (200 → −56).
- The backbone already ends in `Dense(N, softmax)` — do not apply softmax again.
- The three `# PORT` constants select the NPU driver stack (see
  [architecture → same NPU, two drivers](architecture.md#same-npu-two-drivers)).

## classifier.py

CPU-side head. The backbone turns a window into a 128-d embedding; the head
classifies it by cosine similarity against per-class prototypes. Pure numpy.

Persisted as JSON next to the backbone (`classifier_head.json`):

```json
{ "labels": ["circle","shake","steady"], "prototypes": [[...128 floats...], ...] }
```

Each label also gets a colour slot `0..PALETTE_SIZE-1`, re-assigned on every
train, so a retrain is visually obvious by design. `save()` writes to a temp file
then renames, so a partial write can't corrupt the head the worker is about to
reload.

## trainer.py

For each selected label: run every `(WINDOW, 3)` window through the backbone,
average to one 128-d prototype, write all to `classifier_head.json`, hot-reload
the live head. No TF, no gradients, seconds. Singleton + background thread +
status pattern mirrors recorder.py.

- `TRAIN_HOP = WINDOW_SIZE // 4` (75 % overlap) → 4 phase alignments per gesture
  cycle, ~4× the samples of non-overlapping slicing, lower-variance prototype.
  50 % overlap only hits 2 phase positions here (`2604/1302 = 2`) — worse.
- Colour assignment uses Python's RNG (not numpy's), so external numpy seeding
  can't stabilise the shuffle.

## recorder.py

Captures raw XYZ to binary alongside live inference. Two-stage queue: `feed()`
(from W1) appends to a stage-1 list under a brief lock, no I/O; once `FLUSH_SIZE`
samples accumulate, the chunk goes to a writer queue drained by a dedicated
thread that flushes ~1×/s. `ndarray.tofile` releases the GIL during the write —
the reason for the binary format; pure-Python CSV formatting held the GIL for the
whole chunk and starved the reader's Modbus poll.

- Binary format: raw float32 little-endian, 3 channels (x, y, z) interleaved, no
  header, `.bin`. Sample count = `file_size // 12`. Append = `open("ab")`.
- Legacy `.csv` files remain readable by trainer.py; new recordings go to `.bin`.
- `delete_label` does **not** re-sanitize the name (that would mangle e.g.
  leading underscores) — it matches what `list_existing_labels` returns, rejects
  path separators, and confirms the resolved path sits directly in `data_dir`.
- `cancel()` hands the writer the leftover stage-1 buffer, then blocks up to
  `WRITER_JOIN_TIMEOUT_S` so `last_finished` reports the true on-disk count.
- **Known open bug:** the recorder produces a periodic ~9400-sample seam in
  `data/*.bin`. The reader is healthy (`data_len` stable, `total_fails=0`).
  Suspect `_writer_queue` / status-poll lock contention.

## state.py

Shared primitives for all three workers.

- `load_class_labels()` / `CLASS_LABELS` is a startup hint only (templates,
  colour map). The authoritative class name per prediction comes from the head
  live at invoke time, so a stale value here is fine.
- `SnapshotBus` wakes any number of SSE clients on each latch. Each latch bumps a
  monotonic counter and notifies all waiters; SSE generators track their
  last-seen counter so no client misses a notification.
- `RollingPredictions` keeps the last N results and a displayable snapshot that
  refreshes every `DISPLAY_REFRESH_EVERY` pushes. Majority class is tie-broken by
  recency (walk the deque in reverse). The `on_latch` callback fires **outside**
  the lock (`bus.bump()` takes its own lock). `clear()` runs when switching the
  active sensor so stale predictions don't skew the new majority.

## bin2csv.py

Convert recorder `.bin` (raw float32 XYZ, 3 channels interleaved, no header) to
`.csv` with an `x,y,z` header. No timestamps — the `.bin` format never stored
them.

```bash
python bin2csv.py data/steady.bin data/shake.bin   # specific files
python bin2csv.py data/*.bin                        # shell glob
python bin2csv.py data                              # every .bin in a dir
```
