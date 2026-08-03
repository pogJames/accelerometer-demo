# Pages

## Live waveform `/`

2 × 2 grid of per-sensor uPlot canvases.

- **Raw mode** — scrolling time-domain trace, X/Y/Z overlaid, "Latest N samples"
  configurable up to 1 s, Refresh rate selectable 10/30/60 fps.
- **FFT mode** — magnitude spectrum, FFT range configurable in Hz (caps at
  800 Hz).
- Per-axis stat chips above each plot (min..max in raw, peak frequency in FFT).
  Click a chip to hide/show that axis.
- Hover the plot for an inline tooltip with the sample value at that point.

## Live inference `/inference`

One accordion card per sensor. Shows majority class label over the last 7
results, latest confidence, and a colour-coded dot strip. SSE-driven, no
polling.

## Metrics data `/metrics`

Per-sensor card showing temperature + gravity table (RMS / Peak / Crest /
Skewness / Kurtosis × XYZ + primary freq) + velocity table (RMS / Peak / Crest ×
XYZ + primary freq). The reader polls kurtosis at ~2 Hz and emits a fresh
**batch** of all metrics only when kurtosis changes (or 5 s fallback) — so every
field on the page advances together at the slow ~2–5 s cadence.

Activating `/metrics` does **not** start raw streaming on a port — FC03 polling
only. The `/` and `/inference` pages activate raw separately; raw streaming
always preempts metric polling on the same port.

## Data recording `/record`

Records raw XYZ samples to `data/<name>.bin` (float32 little-endian, 3 channels
interleaved). Each row in the "Existing data" table has a trash-icon delete
button (confirmation prompt). Append / Overwrite modes; progress streams over
SSE.

> To convert a recorded `.bin` to `.csv` (`x,y,z`, no timestamps), run the
> converter manually: `python bin2csv.py data/<name>.bin` (also accepts a glob
> or a directory).

## Train model `/train`

"Training" = computing one mean embedding per label. No gradients, takes
seconds. Select ≥ 2 eligible labels → click **Train model** →
`classifier_head.json` written → `InferenceWorker` hot-reloads with no app
restart. Colour slots are randomly re-assigned on each retrain.

## Settings `/settings`

One row per detected port; type a friendly alias (e.g. "Front-left motor") and
it's saved instantly to `port_aliases.json`. Aliases render across every page
that shows a port name; the raw `/dev/ttyUSBx` paths remain the internal IDs.
