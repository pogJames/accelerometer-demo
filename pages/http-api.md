# HTTP API

All bodies are JSON. Errors return `{"error": "..."}` with a 4xx/5xx status.
Ports are the raw `/dev/ttyUSBx` strings.

## SSE streams

Long-lived `text/event-stream` connections; each `data:` frame is one JSON
snapshot, pushed on change (heartbeat comment on idle). A stream is inert until
a page activates the relevant port — see [pages.md](pages.md).

| Endpoint | Push trigger | Frame payload |
|---|---|---|
| `GET /stream/inference` | new majority-vote latch | `{ports:{<port>:{majority_class_name, majority_count, window_count, recent[], display_seq, latest_ts}}, class_labels[], label_colors{}, active_ports[], inference_mode, now}` |
| `GET /stream/waveform` | 30 Hz display tick | `{ports:{<port>:{raw, fft, raw_seq, fft_seq, ts}}, raw_axis[], freq_axis_hz[], sample_rate, window_size, fft_bins, …}` |
| `GET /stream/metrics` | FC03 batch (~2–5 s) | `{ports:{<port>:{temperature, gravity{rms,peak,crest,skewness,kurtosis,primary_freq}, velocity{rms,peak,crest,primary_freq}, ts}}, open_ports[], now}` |

## Activation & config (POST)

| Endpoint | Request body | Response | Effect |
|---|---|---|---|
| `POST /api/active_port` | `{port, active: bool}` | `{port, active}` | Open/close a port for **raw + inference** streaming |
| `POST /api/metrics_active` | `{port, active: bool}` | `{port, active}` | Open/close **FC03 metric** polling (independent of raw) |
| `POST /api/waveform_config` | `{fft_max_hz?, raw_samples?}` | applied values (`{fft_max_hz, fft_bins, raw_samples}`) | Live waveform/FFT knobs |
| `POST /api/port_alias` | `{port, alias}` | `{port, alias}` | Set a port's friendly display name |

## Recording

| Endpoint | Request body | Response |
|---|---|---|
| `GET /api/recordings` | — | `{labels:[{name, samples, windows, eligible}], ports[], sample_rate, min_samples, min_windows, window_size, data_dir}` |
| `GET /api/record/status` | — | `{session: {status, name, port, progress, elapsed_s, samples_written, target_samples, file_path} \| null}` (client-polled ~1 Hz) |
| `POST /api/record/start` | `{name, target_samples, port, mode:"append"\|"overwrite"}` | `{session}` |
| `POST /api/record/cancel` | — | `{session}` (keeps samples written so far) |
| `POST /api/recordings/delete` | `{name}` | `{deleted}` — `409` if that label is recording |

## Training

| Endpoint | Request body | Response |
|---|---|---|
| `GET /api/train/status` | — | `{session, head_labels[], head_colors{}, backbone, backbone_mode}` |
| `POST /api/train/start` | `{labels:[...]}` (≥ 2 eligible) | `{session}` |
| `POST /api/train/cancel` | — | `{session}` |

## One-shot snapshots (JSON)

Non-streaming GETs returning the same payload as the matching `/stream/*` feed —
for `curl`/`jq` debugging without holding an SSE connection open. Reflect the
current cached state regardless of whether any port is activated.

| Endpoint | Response |
|---|---|
| `GET /api/inference` | Current `/stream/inference` payload |
| `GET /api/metrics` | Current `/stream/metrics` payload |
| `GET /api/waveform` | Current `/stream/waveform` payload |
