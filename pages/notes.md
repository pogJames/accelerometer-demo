# notes.md

Per-module design rationale extracted from source comments — the long-form
"why" the code points to with `See pages/notes.md`. For usage-level docs see
[README.md](../README.md) and its siblings: [architecture.md](architecture.md),
[http-api.md](http-api.md), [pages.md](pages.md), [tuning.md](tuning.md).

## Three-worker architecture

```
W1 sensor_reader.reader_process_main  (one subprocess per port)
     → window_queue → W2 inference.InferenceWorker (one thread)
                        → RollingPredictions → W3 app.py / Flask → browser (SSE)
```

- **W1** reads Modbus, emits `(WINDOW_SIZE, 3)` windows for inference and
  small raw chunks for the live waveform. Runs in its own process because the
  serial poll loop is GIL-bound — sharing the GIL with Flask / InferenceWorker
  / GC pauses lets the sensor's hardware FIFO (max 65535 regs ≈ 2.8 s at
  7812 Hz) overflow and silently drop samples. The subprocess also owns the
  `RecordingManager`; raw samples never cross the IPC boundary. Control plane
  (start/cancel/status) is RPC over a request/response `mp.Queue` pair drained
  by a sibling thread.
- **W2** runs the frozen backbone + cosine-sim head. Classify only — the
  waveform path is fed by its own `raw_queue` + drain thread in app.py, so W2
  is a single get→classify→push loop and doesn't compete with it.
- **W3** wires everything and serves the dashboard over SSE.

## inference.py

Each `(WINDOW, 3)` window → backbone → 128-d embedding → `head.predict()` →
`(class_id, similarity, class_name)`. Trained instantly from `data/*.bin` (or
legacy `*.csv`) via trainer.py. `"untrained"` and `"stub"` states let the
dashboard render before artifacts exist.

Backend: NXP i.MX Ethos-U NPU (Vela-compiled int8 model through
`tflite_runtime` + the Ethos-U delegate) — the only supported backend. If the
runtime, model, or delegate is missing, or the delegate fails to load, it drops
to `"stub"` (class 0, confidence 1.0) so dev machines and the dashboard still
run. **There is no CPU fallback** in this build.

- `embed()` is public so trainer.py can borrow the interpreter to compute
  prototypes; it holds `_invoke_lock` for the TFLite call.
- `reload_head()` atomically swaps the head from disk so a retrain surfaces new
  labels without a process restart.
- **Clip-before-cast** when quantizing input: `np.clip(np.round(x/scale + zp),
  -128, 127).astype(np.int8)`. A bare `.astype` wraps out-of-range values
  (200 → −56) and feeds the NPU garbage.
- The model already has `Dense(N, softmax)` baked in — do not apply softmax
  again in the classify path.

### Porting to another NPU

Single-path build targeting NXP i.MX / Ethos-U; all knobs are in `inference.py`,
marked `PORT:`. Retarget = change three things:

| What | `inference.py` | NXP (default) | Matrix800 / VeriSilicon |
|---|---|---|---|
| Runtime import | `_try_load_interpreter` `# PORT` | `tflite_runtime.interpreter` | `ai_edge_litert.interpreter` |
| Delegate `.so` | `DELEGATE_PATH` | `/usr/lib/libethosu_delegate.so` | `/usr/local/lib/aarch64-linux-gnu/libteflon.so` |
| NPU model | `NPU_MODEL_PATH` | `..._int8_vela.tflite` | `..._int8.tflite` (non-Vela) |

A float32 `.tflite` loaded via a VeriSilicon-style delegate (e.g.
`libteflon.so`) does not error and does not warn — it returns bit-identical
garbage for every input. Verify with `smoketest.py` first; set `FORCE_CPU=1`
while debugging.

## classifier.py

Lightweight CPU-side head. The backbone turns a `(WINDOW, 3)` window into a
128-d float embedding; the head classifies it by cosine similarity against
per-class prototypes. Pure numpy, no TF.

Persisted as JSON next to the backbone (`classifier_head.json`):

```json
{ "labels": ["circle","shake","steady"], "prototypes": [[...128 floats...], ...] }
```

"Training" = computing one mean embedding per label (done by trainer.py). Each
label also gets a colour slot `0..PALETTE_SIZE-1`; slots are re-assigned on
every train so a label can change colour between runs — by design, so a retrain
is visually obvious. `save()` writes to a temp file + rename so a partial write
can't corrupt the head the running worker is about to reload.

## state.py

Shared primitives for all three workers.

- `load_class_labels()` / `CLASS_LABELS` is a startup hint only (templates,
  colour map). The authoritative class name per prediction comes from the head
  live at invoke time (`InferenceWorker` reads `self.head.labels` directly), so
  a stale snapshot here is fine.
- `SnapshotBus` wakes any number of SSE clients on each latch. Each latch bumps
  a monotonic counter and notifies all waiters; SSE generators track their
  last-seen counter so they don't miss notifications with multiple clients.
- `RollingPredictions` keeps the last N results and an event-driven displayable
  snapshot that only refreshes every `DISPLAY_REFRESH_EVERY` pushes. Majority
  class is tie-broken by recency (walk the deque in reverse). The `on_latch`
  callback fires **outside** the lock because `bus.bump()` takes its own lock.
  `clear()` is called when switching the active sensor so stale predictions
  don't influence the new majority.

## sensor_reader.py

Streams raw XYZ from input register `0x02` (FC04) and emits `WINDOW_SIZE`
sliding windows (`HOP_SIZE` hop) to the inference queue. Read pattern mirrors
`DAQ_Modbus_MultiChs_v1.3.py`; queues use drop-oldest.

`reader_process_main` arg contract:
- `port` — e.g. `/dev/ttyUSB0`.
- `window_queue` — push `(port, np.ndarray)` for inference.
- `req_q` / `resp_q` — RPC in / responses out.
- `stop_event` — request shutdown.
- `active_event` — main process sets it when the port should read raw. When
  clear the reader sleeps and burns no Modbus traffic. Each inactive→active
  transition re-writes the sample-rate register, which the sensor treats as a
  FIFO reset so the first window after resume is fresh.
- `data_dir` — where recordings get written.

Process/scheduling notes:
- `spawn` gives the child a fresh interpreter, so it re-applies the
  `site-packages` path before importing project modules.
- No CPU pinning; the kernel schedules readers across cores. `SCHED_FIFO`
  priority 50 still preempts normal-priority processes when a reader has work
  (needs root / `CAP_SYS_NICE`; logged and skipped otherwise).

`READ_TIMEOUT_S = 0.05`: 50 ms is well above the sensor's response time at
3 Mbaud (request ~30 µs, processing a few ms, response ~700 µs — normally
<10 ms) but turns a bad-packet stall into a ~100 ms blip. If `total_fails`
climbs during normal operation, bump it — the sensor's tail latency may be
longer.

Computed-metric registers (FC03) **alias**: each metric is its own
`read_holding_registers(base, count=3)` — a contiguous block read returns
garbage. Addresses + scaling mirror the validated Rust client
(`src/types.rs`, `src/modbus.rs`).

Metric polling: kurtosis (slowest metric) is polled at `KURT_POLL_INTERVAL_S`
and a full batch is emitted only when it changes (or first poll, or after
`METRIC_FALLBACK_S`), so the whole set updates together at the slow 2–5 s
cadence instead of spamming the fast metrics. FC03 reads are genuinely slow +
variable on this sensor (~0.5–1 s); a premature timeout leaves the slow
response in flight and the next request reads that stale reply ("wrong byte
count") — desyncing the stream forever. Hence `METRIC_READ_TIMEOUT_S = 5.0`
(the Rust client uses the same) swapped in only around metric reads, plus a
line drain to resync. A full ~12-metric sweep therefore has a several-second
floor and can't be batched away (see aliasing above). Raw streaming always
takes priority: `_MetricsAbort` bails a slow sweep the moment raw is requested
so a page switch to the waveform view isn't blocked behind FC03.

`skewness` can be negative on the wire — currently read unsigned; if it reads
wrong, view as int16 (flagged for the hardware-validation pass).

## recorder.py

Captures raw XYZ to binary alongside live inference. Two-stage queue: `feed()`
(from W1) appends to a stage-1 list under a brief lock, no I/O; once
`FLUSH_SIZE` samples accumulate the chunk goes to a writer queue drained by a
dedicated thread that flushes ~1×/s. `ndarray.tofile` releases the GIL during
the bytes-to-disk handoff — the whole point of the binary format; pure-Python
CSV formatting (the old approach) held the GIL for the entire chunk and starved
the reader's modbus poll.

Binary format: raw float32 little-endian, 3 channels (x_g, y_g, z_g)
interleaved, no header, `.bin`. Sample count = `file_size // 12`. Append mode is
just `open("ab")`. Legacy `.csv` files remain readable by trainer.py; new
recordings always go to `.bin`.

`delete_label` does **not** re-sanitize the name (that would mangle e.g.
leading underscores) — it matches what `list_existing_labels` returns, rejects
path separators, and confirms the resolved path sits directly in `data_dir`.

`cancel()` hands the writer the leftover stage-1 buffer, then blocks up to
`WRITER_JOIN_TIMEOUT_S` so `last_finished` reports the truthful on-disk count.

**Known open bug:** recorder produces a periodic ~9400-sample seam in
`data/*.bin`. Reader is healthy (`data_len` stable, `total_fails=0`). Suspect
`_writer_queue` / status-poll lock contention.

## trainer.py

For each selected label: run every `(WINDOW, 3)` window through the backbone,
average the embeddings to one 128-d prototype, write all to
`classifier_head.json`, and hot-reload the live head. No TF, no gradients,
seconds. Singleton + background thread + status pattern mirrors recorder.py.

`TRAIN_HOP = WINDOW_SIZE // 4` (75 % overlap) → 4 distinct phase alignments per
gesture cycle, ~4× more training samples than non-overlapping slicing. Lower
variance on the prototype mean + broader phase coverage so the prototype matches
inference-time windows regardless of where the live gesture starts in the
window. 50 % overlap is tempting (2× compute) but `2604/1302 = 2`, so it hits
only 2 phase positions — strictly worse than 25 % overlap here.

Colour assignment is random per train using Python's RNG (not numpy's), so
external numpy seeding can't make the shuffle stable. ≤ `PALETTE_SIZE` labels
get distinct slots; beyond that they wrap.

## waveform.py

Per-port live-waveform aggregator. Fed by a fast raw-sample stream
(sensor_reader ~217-sample chunks at ~36 Hz/port → app.py drain thread →
`append(port, chunk)`) into a per-port circular buffer. A ~30 Hz display tick
calls `render_tick()`, which smooths + decimates the raw view every tick
(smooth scroll) and recomputes the FFT ~every `FFT_INTERVAL_S` (~6 Hz), storing
a per-port snapshot. `snapshot()` is a cheap pure read for the SSE handlers.
Because the ring advances under a large overlapping window, the raw view
scrolls smoothly instead of flashing discrete hops, and "Latest N samples" can
span up to ~1 s.

FFT uses pyfftw with cached FFTW plans (NEON SIMD on aarch64), falling back to
`numpy.fft.rfft` if pyfftw isn't installed.

`WINDOW_SIZE` here is the **FFT window only** — independent from the model's
input window (`WINDOW_SIZE` in sensor_reader.py); do not unify them.

Savitzky-Golay coefficients (`_savgol_coeffs`) are derived with zero scipy
deps: build the Vandermonde matrix `A[i,k] = i**k` for `i in [-half, +half]`,
take row 0 of `pinv(A)` — the coefficients that recover the centre value as a
least-squares polynomial fit. Equivalent to `scipy.signal.savgol_coeffs(W, P)`.
`_savgol` pads edge-replicated so the ends aren't pulled toward zero by
`np.convolve`'s implicit zero-padding.

`WaveformBus` is kept separate from `state.SnapshotBus` so prediction and
waveform SSE don't false-wake each other when only one has new data.

## app.py

Worker 3 — Flask dashboard and the entry point that wires W1/W2/W3. Adds over
project3: `/train` page + train endpoints, a per-port reader subprocess owning
the local `RecordingManager` (main process talks to it via `RemoteRecorder`
RPC), and a `TrainerManager` that borrows the InferenceWorker's interpreter to
compute prototypes then hot-reloads the head. Run: `python app.py` →
`http://localhost:8000/` (binds `0.0.0.0`; `port=80` in `__main__`).

- The `site-packages` path insert lets the embedded build ship bundled deps
  (pymodbus historically). Harmless when empty; kept as a safety net. On Linux
  (fork) children inherit `sys.path`; on Windows (spawn) `reader_process_main`
  re-applies it.
- Queue caps: window pickling is ~1–2 ms for a `(2604,3)` float32 window vs the
  167 ms hop interval, so cap 16 gives 4 readers headroom before `put()` would
  block — drop-oldest kicks in first. Metrics emit every ~2–5 s (small cap
  fine); raw chunks ~144/s across 4 ports (cap 64).
- Reader gating: a reader runs if the user opened the port for inference **or**
  a recording is in progress. `inference_open` / `recording_ports` /
  `metrics_open` are tracked separately; `active_events` (raw) is the union of
  the first two and `metrics_events` (FC03) tracks the third — independent so
  `/metrics` can poll without paying the 3 Mbps raw-stream cost. Recording on a
  port the user never opened for inference must not leave it showing "open" once
  the recording ends, hence `begin_recording` / `end_recording` don't touch
  `inference_open`.
- `record_state["port"]` is set on a successful start and never cleared so
  `/status` keeps reporting the right subprocess's `last_finished` snapshot.
- Recording auto-stops in the subprocess when the target is reached; the main
  process only learns via a status poll (`release_if_finished`).
- One-shot `/api/*` GET snapshots are non-streaming siblings of the `/stream/*`
  SSE feeds, for curl/jq debugging without holding a connection open.

## bin2csv.py

Convert recorder `.bin` (raw float32 XYZ, 3 channels interleaved, no header) to
`.csv` with an `x,y,z` header. No timestamps — the `.bin` format never stored
them.

```
python bin2csv.py data/steady.bin data/shake.bin   # specific files
python bin2csv.py data/*.bin                        # shell glob
python bin2csv.py data                              # every .bin in a dir
```

## smoketest.py

Verify the backbone `.tflite` in isolation and measure pure invoke latency — no
sensor / Flask / multiprocessing. Catches (1) silent NPU corruption: push
zeros/ones/random and confirm outputs actually differ; (2) NPU vs CPU latency:
time N invokes (min/mean/p50/p95/max), optionally both backends.

```
python smoketest.py                   # NPU (fall back to CPU)
FORCE_CPU=1 python smoketest.py       # CPU only
python smoketest.py --runs 500        # more samples for stable timing
python smoketest.py --compare         # run both NPU and CPU, print both
```
